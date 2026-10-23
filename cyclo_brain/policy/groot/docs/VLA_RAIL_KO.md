# 일반 Async용 VLA-RAIL

## 구현 수식과 실행 경로 — 2026-10-21 소스 기준

이 절은 `/home/robotis-ai/cyclo_intelligence`의 현재 작업 트리를 직접 읽어 작성했다. 확인 시 HEAD는 `2091ad06d176ed8d533e51cfe05117d1c5e8b5c3`이다. HEAD만으로 커밋되지 않은 파일 상태까지 재현할 수는 없다. 뒤쪽의 과거 테스트·배포 기록은 기존 기록이며, 이번 문서 작성에서 실기 테스트나 서버 재시작을 다시 수행했다는 뜻이 아니다.

**현재 Async용 VLA-RAIL은 일반 GR00T가 예측한 청크를 전역 cubic으로 근사하고, 현재 명령 상태와 TCP 경로가 가까운 지점을 찾아 C² 연결 곡선으로 교체하는 실행기다.** 별도 RAIL 학습, TTRTC prefix conditioning, MLP/RL 잔차 보정, 속도·가속도 clipping을 수행하지 않는다. 아래의 비용 함수는 학습 loss가 아니라 매 청크 도착 시 후보 곡선을 선택하는 런타임 점수다.

### 1. 모듈별 책임과 데이터 흐름

경로는 저장소 루트 기준이다.

| 소스 | 실제 역할 |
|---|---|
| `cyclo_brain/policy/common/runtime/main_runtime/control_loop.py` | Async/RAIL 선택, URDF FK 초기화, tick·피드백·발행·오류 정지 |
| `cyclo_brain/policy/common/runtime/main_runtime/rail_execution.py` | 비동기 요청 하나 유지, 센서 시간 검증, 청크 수용, 미래 명령 창 생성 |
| `cyclo_brain/policy/common/runtime/main_runtime/rail_trajectory.py` | cubic 최소제곱, quintic/dual-quintic, head/body/tail과 그리퍼 샘플링 |
| `cyclo_brain/policy/common/runtime/main_runtime/rail_alignment.py` | 현재 명령 TCP에 가까운 새 cubic의 연속 위상 탐색 |
| `cyclo_brain/policy/common/runtime/main_runtime/rail_handoff.py` | 연결 후보 생성, 비용 계산, TCP 후퇴·관절 왕복 검사, 수용/거부 |
| `cyclo_brain/policy/common/runtime/main_runtime/rail_kinematics.py` | 로봇 URDF에서 양팔 또는 단팔 FK 계산 |
| `cyclo_brain/policy/common/runtime/main_runtime/rail_output_guard.py` | 수치·차원·초기 피드백·명령 간격 검증; 목표를 제한하지 않음 |
| `cyclo_brain/policy/groot/runtime/inference_engine.py` | 센서 snapshot → 일반 `policy.get_action()` → 원본 청크와 timing 반환 |
| `cyclo_brain/sdk/robot_client/robot_client/robot_client.py` | 센서 원자적 복사, q/v/a JointTrajectory 전송 |

```text
LOAD(async, vla_rail_enabled=true)
  → action_keys·원본 horizon 검증 → URDF FK 준비
RESUME
  → 별도 요청 스레드: snapshot → GR00T → 원본 청크 → cubic fit → pending
  → 정책 tick: pending의 최신성 검사 → TCP 위상 탐색 → 연결 후보 평가
       수용: 기존 참조를 새 head/body/tail로 교체
       거부: 기존 참조 계속 실행 → 유효 구간 끝에서 정지·유지
  → 계획 계산 뒤 현재 시각으로 다시 샘플링
  → q/v/a 미래 창 발행 → 다음 tick 반복
  → 완료된 요청 뒤에는 다음 추론 요청 계속 진행
```

추론 스레드와 명령 발행은 분리하지만 동시 추론 요청을 여러 개 쌓지는 않는다. cubic fitting은 발행 잠금 밖에서 수행한다. handoff 계획은 tick 경로에서 수행하므로 계산 시간이 발행 지터에 영향을 줄 수 있다. 결과를 수용한 시각과 실제 명령 샘플 시각도 구분한다.

### 2. 기호·차원·서로 다른 시간축

| 기호 | 의미 / 코드 |
|---|---|
| \(N\) | 원본 청크의 액션 수, 최소 4 |
| \(f_p\), \(\Delta=1/f_p\) | `inference_hz`, 정책 샘플 간격 |
| \(H=(N-1)\Delta\) | cubic의 정책 시간 horizon |
| \(f_o=f_p\) | 실제 명령 갱신 목표 주파수 `output_hz` |
| \(B\) | 기본 연결 시간 `blend_s=0.2`초 |
| \(\alpha=f_o/f_{interp}\) | 설정 재생 배율; `interpolation_hz` 미지정이면 1 |
| \(t_s,t_r,t_a,t_0\) | 센서 기준 수신 시각, 요청 시작, 응답 도착, 교체 판단 시각 |
| \(\tau\) | 새 청크 안의 정책 시각, \(0\le\tau\le H\) |
| \(\tau_*\), \(\beta\) | 선택된 팔 offset과 실제 재생 배율 |
| \(s=t-t_0\) | 새 실행 궤적 시작 이후 실제 경과 시간 |

선택된 팔 하나당 액션은 8차원이다. 앞 7개가 팔 관절, 마지막 1개가 그리퍼다. 양팔이면 16차원 중 팔 14차원만 다항식/FK/연결 비용에 들어간다. `arm_left`, `arm_right`의 입력 순서를 보존하며 중복이나 다른 그룹은 허용하지 않는다. 이하 \(q\)는 그리퍼를 제외한 관절각 벡터이고 단위는 rad이다.

예를 들어 \(N=16,f_p=15\)이면 \(H=1\)초다. 마지막 샘플의 시각이 15/15초이므로 16/15초가 아니다. `inference_hz=15`는 GPU가 초당 15번 추론한다는 뜻이 아니며, 일반 `control_hz`는 RAIL의 발행 주기와 재생 배율을 바꾸지 않는다.

### 3. 전체 청크를 cubic으로 근사

원본 관절 액션을 \(q_i\), 정규화 시간을 \(u_i=i/(N-1)\)라 두면:

\[
V_{i,:}=[1,u_i,u_i^2,u_i^3],\qquad
C=V^+Q=\underset{C}{\operatorname{argmin}}\;\|VC-Q\|_F^2.
\]

`CubicChunk`는 `np.linalg.pinv`를 사용한다. 역행렬 \((V^TV)^{-1}\)을 직접 계산하지 않으며, 같은 N의 fit 행렬은 캐시한다. 모든 원본 샘플을 사용하고 fitting 전에 오래된 prefix를 잘라내지 않는다.

\[
p(\tau)=c_0+c_1u+c_2u^2+c_3u^3,\qquad u=\tau/H,
\]
\[
p'(\tau)=\frac{c_1+2c_2u+3c_3u^2}{H},\quad
p''(\tau)=\frac{2c_2+6c_3u}{H^2},\quad
p'''(\tau)=\frac{6c_3}{H^3}.
\]

이는 원본 점을 모두 통과하는 보간이 아니라 **최소제곱 근사**다. 고주파 왕복이 줄어들 수 있지만 세밀한 작업 동작도 사라질 수 있다. 처음/마지막 원본 값의 정확한 보존, 원본 min/max 안에 머무름, 절대 jerk 상한은 보장하지 않는다. production은 `enforce_envelope=False`이며 위치 clipping을 하지 않는다.

### 4. 관측 지연은 고정 상수가 아니라 수신 시각으로 계산

\[
A=\max(0,t_0-t_s),\qquad \tau_{nom}=\alpha A.
\]

`get_observation_snapshot()`은 선택된 이미지·관절과 콜백 수신 monotonic 시각을 같은 잠금에서 복사한다. \(t_s\)는 선택 관절 중 가장 오래된 수신 시각이다. 관절을 선택하지 않았다면 선택 이미지의 가장 오래된 시각을 쓴다. 이미지의 더 오래된 시각이 있더라도 관절이 있으면 관절 기준이며, 각 이미지 age/skew 검사는 별도로 수행한다.

`clock_id`(Linux boot ID)가 같고 시각/출처가 유효하면 그 센서 시각을 사용한다. 없거나 잘못되면 요청 시작 \(t_r\)로 대체한다. 그러면 \(A=t_0-t_r\)가 되어 요청 전 센서 캐시 나이는 포함하지 못한다. 이 fallback도 매번 실측 시각 차를 계산하며 지연을 하드코딩하는 방식이 아니다.

유효한 시각을 사용할 때 다음 분해가 가능하다.

\[
t_0-t_s=(t_r-t_s)+(t_a-t_r)+(t_0-t_a).
\]

첫 항은 시간 순서에 따라 음수일 수도 있다. 센서 snapshot이 요청 후 갱신될 수 있기 때문이다. 따라서 이를 항상 양수인 세 개의 독립 지연 측정치로 해석하면 안 된다. 전체 \(A\)에는 전처리·추론·전송·응답 이후 대기 등이 포함될 수 있으나 카메라 노출→콜백 수신 지연은 직접 측정하지 않는다. 즉 완전한 물리 센서 지연 \(\Delta t_o\)를 측정한 것은 아니다.

\[
\tau_{nom}\ge H \quad\Longrightarrow\quad \texttt{expired\_horizon}:\ \text{새 청크 거부}.
\]

유효한 \(\tau_{nom}\)은 **팔 정렬 offset의 하한이 아니다**. 팔은 공간 위상으로, 그리퍼는 관측 시간 기준으로 정렬한다.

### 5. 현재 명령과 가까운 TCP 위상 탐색

교체 시점의 이전 명령 궤적을 \(r_{old}\)라 하면:

\[
(q_0,v_0,a_0)=\big(r_{old}(t_0),\dot r_{old}(t_0),\ddot r_{old}(t_0)\big).
\]

현재 명령 상태를 기준으로 연결하므로 매번 지연된 실측 q로 명령을 되돌리지 않는다. 최초 시작만 최신 피드백 q와 \(v_0=a_0=0\)을 쓴다. 이는 실제 팔이 반드시 정지했다는 측정 결과가 아니라 bootstrap 경계 조건이다.

선택된 팔 \(a\)의 TCP FK를 \(F_a\)라 할 때:

\[
\tau_{near}\approx\underset{0\le\tau\le H}{\operatorname{argmin}}
\sum_a\|F_a(p(\tau))-F_a(q_0)\|_2^2.
\]

URDF의 `base_link`→`end_effector_l_link`/`end_effector_r_link`를 사용한다. lift는 0으로 두며 비교되는 두 자세에서 동일한 lift 평행 이동이 상쇄된다는 가정이다. 실제 world TCP/접촉/충돌을 추정하지 않는다. 자세 방향은 이 거리식에 직접 포함하지 않고, 뒤의 관절 공간 비용으로 간접 고려한다.

구현은 65개 coarse 시점, 비용이 낮은 내부 국소 최소 최대 8개와 양끝 구간을 대상으로 25회 golden-section 탐색을 수행한다. 전역 최적해의 엄밀한 증명은 아니다. TCP 비용이 수치적으로 같은 경우에만 다음 작은 관절 tie-break를 더한다.

\[
C_{order}=C_{TCP}+10^{-14}\frac{J}{1+J},\qquad J=\|p(\tau)-q_0\|^2.
\]

가장 가까운 시점 외에 \(\pm\Delta,\pm\Delta/2\), 다른 basin, \(H-\Delta\)에서 제한된 대안을 구성한다. 원본 cubic의 coarse TCP 경로 길이를 \(L_a\)라 하면 production에서 사용하는 근접 허용량은:

\[
\sqrt{C_{TCP}(\tau)}\le\sqrt{C_{TCP}(\tau_{near})}
+\sqrt{\sum_a(0.005+0.02L_a)^2}.
\]

가까운 대안 최대 3개를 남긴다. `nearest_tcp_phases()`의 독립 기본 인자는 0.003m이지만 **production `select_handoff()`는 0.005m을 전달한다**. 최근접이 끝점이면 미래가 없다는 사실을 유지하고 가까운 대안을 검사한다. 나이 기반 먼 offset으로 강제 점프하는 fallback은 없다.

### 6. 다음 응답까지 경로가 유지되도록 시간 재조정

요청 왕복 시간 \(R_k=t_a-t_r\)에 대해 유효 horizon 응답만 다음 budget을 갱신한다.

\[
b_k=\max(R_k,0.9b_{k-1}),\qquad
T_{min}=\max(2B,b_k+B+2/f_o).
\]

이는 최근 지연 증가에는 즉시 반응하고 감소에는 완만히 반응하는 실행 시간 확보 값이다. 실시간 deadline 보장은 아니다. 후보 offset \(\tau_*\)마다:

\[
\beta=\min\left(\alpha,\frac{H-\tau_*}{T_{min}}\right),\quad
T=\frac{H-\tau_*}{\beta},\quad \tau(s)=\tau_*+\beta s.
\]
\[
q_{ref}(s)=p(\tau(s)),\quad
v_{ref}(s)=\beta p'(\tau(s)),\quad
a_{ref}(s)=\beta^2p''(\tau(s)).
\]

body의 jerk는 \(\beta^3p'''\)로 바뀐다. 이 감속은 다음 청크를 기다리는 시간을 확보하기 위한 retiming이며 관절별 속도 제한이 아니다. 이전 연결 경계의 \(v_0,a_0\)에는 새 \(\beta\)를 다시 곱하지 않는다.

예를 들어 B=0.2초, 15Hz, budget=0.65초라면 \(T_{min}\simeq0.9833\)초다. H=1초, 선택 offset=0.2초, α=1이면 β≈0.8136으로 남은 0.8 정책초를 약 0.9833 실제초에 실행한다.

### 7. C² head/body/tail 곡선

production 후보는 head 시간 \(D_h\in\{B,1.25B,1.5B,2B,2.5B\}\), tail 시간 \(D_t\in\{B,B/2\}\)를 조합한다. \(T\ge D_h+D_t\)인 후보만 평가한다. 기본값이면 head 0.2/0.25/0.3/0.4/0.5초, tail 0.2/0.1초다.

구간 길이 D의 quintic을 \(h(s)=\sum_{k=0}^5 b_k(s/D)^k\)로 두고 시작/끝 q/v/a를 각각 \((q_s,v_s,a_s),(q_e,v_e,a_e)\)라 하면:

\[
b_0=q_s,\quad b_1=Dv_s,\quad b_2=\tfrac12D^2a_s,
\]
\[
d=q_e-b_0-b_1-b_2,\quad e=Dv_e-b_1-2b_2,\quad f=D^2a_e-2b_2,
\]
\[
b_3=10d-4e+\tfrac12f,\quad b_4=-15d+7e-f,\quad b_5=6d-3e+\tfrac12f.
\]

물리 시간 미분은 각각 \(1/D,1/D^2,1/D^3\)을 곱한다. head는 이전 명령 q/v/a에서 \(q_{ref}(D_h),v_{ref}(D_h),a_{ref}(D_h)\)로 연결한다. tail은 \(T-D_t\)의 새 참조 상태에서 \((p(H),0,0)\)으로 연결한다.

\[
r(s)=\begin{cases}
h_{head}(s),&0\le s<D_h,\\
p(\tau_*+\beta s),&D_h\le s<T-D_t,\\
h_{tail}(s-(T-D_t)),&T-D_t\le s<T,\\
p(H),&s\ge T.
\end{cases}
\]

head/body/tail 경계는 위치·속도·가속도가 이어지는 **C²**다. jerk가 경계에서 연속인 C³는 아니며, C²만으로 jerk 크기·후퇴·오버슈트를 억제하지 못한다. 새 청크가 없으면 유한 tail을 마친 후 최종 위치와 0 속도·가속도를 유지한다. 끝점 이후 cubic 외삽은 하지 않는다.

`DualQuintic`도 코드에 남아 있지만 production의 TCP planner는 `head_mode='single'`로 head와 tail 모두 단일 quintic을 쓴다. dual 방식은 중간 q/v를 양끝 평균, 중간 a를 0으로 놓고 길이 D/2 두 구간을 붙인다. 이것을 현재 실행 방식이라고 설명하면 잘못이다.

### 8. 실제 런타임 비용 함수

각 후보에서 head 17점, body 8점, tail 16점, 합계 41개 시각으로 곡선을 평가한다. 같은 시각이 생길 수 있으므로 jerk 계산 시 1ns 이하 간격은 제외한다. 아래 평균은 **시간 적분이 아닌 샘플 산술평균**이다. 구간별 샘플 간격이 달라 head/tail의 비중이 상대적으로 높다.

관절 j의 원본 범위 \(R_j=\max_i q_{ij}-\min_i q_{ij}\), 최근접 위상 \(\tau_n\)을 사용해:

\[
m_j=\max\{R_j,|v_{0j}|B,\tfrac12|a_{0j}|B^2,0.1|p_j(\tau_n)-q_{0j}|\},
\]
\[
M=\max(\max_jm_j,10^{-12}),\quad S_j=\max(m_j,0.05M),
\quad w_j=\frac{m_j^2/(m_j^2+(0.05M)^2)}{\sum_lm_l^2/(m_l^2+(0.05M)^2)}.
\]

구현은 분모를 최소 \(10^{-12}\)로 보호한다. 움직임이 모두 0이면 가중치도 0이다. \(S_j\)는 정규화 척도이고 이동 한도가 아니다.

\(\lambda(s)=\min(1,s/D_h)\)라 두고 이전 참조와 새 cubic을 섞은 비교 목표를 만든다.

\[
q_d=(1-\lambda)q_{old}+\lambda q_{ref},\qquad
v_d=(1-\lambda)v_{old}+\lambda v_{ref}.
\]

이 \(v_d\)는 \(q_d\)를 미분한 값이 아니라 **비용 비교용 속도 혼합값**이다. 실제 명령은 혼합값이 아니라 위 quintic/cubic이다. bootstrap의 비교 참조는 \(q_0+sv_0+\tfrac12s^2a_0\)와 \(v_0+sa_0\)다.

\[
J_q=\sum_jw_j\operatorname{mean}_i\left(\frac{r_{ij}-q_{d,ij}}{S_j}\right)^2,
\quad J_v=\sum_jw_j\operatorname{mean}_i\left(\frac{B(\dot r_{ij}-v_{d,ij})}{S_j}\right)^2,
\]
\[
\hat j_{ij}=\frac{\ddot r_{i+1,j}-\ddot r_{ij}}{s_{i+1}-s_i},\qquad
J_{jerk}=\sum_jw_j\operatorname{mean}_i\left(\frac{B^3\hat j_{ij}}{S_j}\right)^2.
\]

따라서 jerk 항은 analytic jerk 적분이 아니라 가속도 차분 기반 점수다.

관절별 누적 절대 이동량에서 원본 참조의 이동량과 새 경로로 합류하는 데 필요한 거리를 빼고 추가 왕복을 계산한다.

\[
E_{ij}=\frac{\left[\sum_{k<i}|r_{k+1,j}-r_{kj}|-
\sum_{k<i}|q_{ref,k+1,j}-q_{ref,kj}|-|q_{ref,0j}-q_{0j}|\right]_+}{S_j},
\quad E=\max_i\sqrt{\sum_jw_jE_{ij}^2}.
\]

\(E>0.5\)이면 후보를 거부한다. 기본 점수는:

\[
J_{base}=J_q+0.3J_v+0.001J_{jerk}+2E^2
+0.4\left(\frac{\tau_*-\tau_n}{\max(H-\tau_n,\Delta)}\right)^2
+0.015\left(\frac{D_h}{B}-1\right)^2.
\]

이는 위치/속도 충실도, jerk, 추가 왕복, 가까운 위상 유지, 과도한 연결 시간의 절충이다. jerk만 최소화하는 최적제어기는 아니다.

### 9. TCP 추가 후퇴 검사와 최종 후보 선택

이전 TCP 속도를 유한차분으로 계산한다(ε=\(10^{-5}\)).

\[
v_a^{TCP}\approx\frac{F_a(q_0+\epsilon v_0)-F_a(q_0-\epsilon v_0)}{2\epsilon},
\quad d_a=\frac{v_a^{TCP}}{\max(\|v_a^{TCP}\|,10^{-15})}.
\]

이 방향에 대한 음의 이동량을 각 prefix까지 누적한다. 실제 후보 TCP를 \(x_{ia}=F_a(r_i)\), 비교 참조를 \(x^{ref}_{ia}=F_a(q_{ref,i})\)라 하면:

\[
B_{ia}=\sum_{k<i}[-(x_{k+1,a}-x_{ka})\cdot d_a]_+,\quad
B^{ref}_{ia}=\sum_{k<i}[-(x^{ref}_{k+1,a}-x^{ref}_{ka})\cdot d_a]_+,
\]
\[
X_{ia}=[B_{ia}-B^{ref}_{ia}]_+,\quad
L^{ref}_{ia}=\sum_{k<i}\|x^{ref}_{k+1,a}-x^{ref}_{ka}\|,
\quad \eta_{ia}=0.005+0.02L^{ref}_{ia}.
\]

유의하게 이동하는 TCP에서 어느 prefix든 \(X_{ia}>\eta_{ia}+10^{-10}\)이면 거부한다. 이동 여부는 \(\|v_a^{TCP}\|B>\max(10^{-10},0.01L^{ref}_{final,a})\)로 판정한다. 거의 정지한 TCP의 수치 잡음을 의도된 진행 방향으로 간주하지 않는다.

원본 예측에 있는 정상적인 후퇴는 \(B^{ref}\)로 허용한다. 이후의 정상 후퇴가 앞부분의 불필요한 후퇴를 상쇄하지 못하도록 최종 합계만이 아니라 **모든 prefix**를 비교한다. 5mm는 추가 후퇴 허용량이지 위치 도킹 오차나 관절 속도 제한이 아니다.

최종 점수에는 다음이 추가된다.

\[
J=J_{base}+5\operatorname{mean}_a
\left(\frac{\max_iX_{ia}}{\max(0.005+0.02L^{ref}_{final,a},10^{-9})}\right)^2.
\]

비유한 수치, 관절 추가 왕복, TCP 추가 후퇴로 먼저 후보를 걸러낸다. 이후 **위상 후보 순위 우선, 같은 위상 안에서 J가 작은 순서**로 검사한다. 즉 더 먼 위상이 점수가 낮아도 가까운 위상보다 먼저 선택하지 않는다.

각 위상에서 상위 최대 4개 후보만 전체 구간 193점 + head 65점 + tail 65점의 합집합으로 TCP 후퇴를 재검사한다. 통과한 첫 후보를 수용한다. 이 dense 재검사는 TCP 검사를 강화하는 것이며 모든 관절 제약을 연속 시간에서 증명하는 절차가 아니다. 통과 후보가 없으면 `incompatible_bridge`로 거부한다.

### 10. 그리퍼와 팔을 같은 시간축으로 묶지 않는 이유

팔은 공간 최근접 위상으로 정렬하기 때문에 그 offset을 그대로 그리퍼에 쓰면 이미 닫아야 할 그리퍼가 과거 열림 상태로 되돌아갈 수 있다. 따라서:

\[
\tau_g=\begin{cases}0&\text{bootstrap}\\\tau_{nom}&\text{그 이후}\end{cases},\quad
\beta_g=\min\left(\alpha,\frac{H-\tau_g}{T_{min}}\right),
\]
\[
i_g(s)=\operatorname{clip}\left(\left\lfloor\frac{\min(H,\tau_g+\beta_gs)}{\Delta}+10^{-9}\right\rfloor,0,N-1\right),
\quad g(s)=g^{raw}_{i_g(s)}.
\]

그리퍼는 cubic/quintic으로 근사하지 않는다. 팔과 독립된 이산 시간 선택이므로 팔·그리퍼의 실제 작업 타이밍 일치까지 보장하지는 않는다. 미래 전송 창 안에는 현재 g를 동일하게 반복하고 v/a를 0으로 넣어 다음 grasp를 미리 보간하지 않게 한다.

### 11. ROS에 보내는 것은 단일 q가 아닌 짧은 q/v/a 창

출력 주기 \(\Delta_o=1/f_o\)마다:

\[
\mathcal T=\{0,\Delta_o,2\Delta_o,3\Delta_o\}
\cup\{\text{창 안의 head/tail 경계 시각}\}
\]

에 대해 \((r(t+\delta),\dot r(t+\delta),\ddot r(t+\delta))\)를 JointTrajectory로 전송한다. 15Hz에서는 0/66.7/133.3/200ms다. dual-quintic 호환성을 위해 반구간 경계도 포함하며, 단일 quintic에서 그 추가 지점은 곡선을 바꾸지 않는다. 나노초로 반올림한 뒤 너무 가까운 knot를 병합해 짧은 구간의 수치 미분 증폭을 방지한다.

`header.stamp=0`이므로 수신 즉시 시작하는 상대 시각이다. 따라서 **내부 참조의 C² 연속성**, **ROS 메시지의 q/v/a**, **실제 하드웨어 추종의 부드러움**은 구분해야 한다. 전송 지연·제어기 trajectory 교체·모터 설정·기구 유격 때문에 실측 jerk는 달라질 수 있다. 모든 그룹 메시지를 검증한 뒤 발행하지만 양팔에 대한 물리적 원자 전송까지 보장하는 것은 아니다.

### 12. 거부·정지·진단의 해석

| 상황 | 동작 |
|---|---|
| 만료/미래 없음/시간 부족/연결 후보 부적합 | 새 청크만 거부, 기존 유한 참조와 종단 hold 유지, 다음 요청 지속 |
| 추론 요청 오류 | 현재 참조 유지, 약 0.1초 뒤 재요청 가능 |
| 비유한 명령/차원 오류/출력 간격 이상/초기 피드백 실패 | 출력 오류 경로로 추론 중지 및 hold 처리 |
| STOP·CLEAR·설정/지시 변경 | epoch/reset으로 이전 pending과 늦게 온 응답 폐기 |

출력 간격 검사는 \(0<\Delta t\le\max(0.1,3/f_o)\)이다. 15Hz면 0.2초다. 이것은 **속도 제한이 아니라 오래 멈춘 출력 루프 탐지**다.

추종 진단은 \(e=q_{ref}-q_{measured}\)에서 최대 절대값/RMS와 다음 값을 기록한다.

\[
\hat\ell=\frac{e^Tv_{ref}}{v_{ref}^Tv_{ref}}.
\]

\(\hat\ell\)는 국소 진행 방향으로 투영한 오차의 시간 환산값이며 실제 통신 지연 측정치가 아니다. 참조 속도가 거의 0이면 생략한다. 이 값으로 매 청크 offset이나 속도를 직접 수정하지 않는다.

### 13. 논문 비교 코드와 현재 운영 경로의 경계

| 요소 | 현재 production |
|---|---|
| 전체 청크 cubic 최소제곱 | 사용 |
| `alignment_offset()`의 부호 점수 정렬 | 비교용 구현에 남음; TCP 경로에서는 호출하지 않음 |
| `DualQuintic` | 비교/API 호환용; TCP planner는 single quintic 선택 |
| TCP 최근접 위상 + 제한된 대안 | 사용 |
| head/body/tail 후보 비용 및 후퇴 검사 | 사용 |
| 응답 budget 기반 retiming | 사용 |
| TCP provider가 없을 때 `_select_joint_handoff()` | 독립 호출 fallback; production LOAD는 FK 필수라 조용히 이 경로로 우회하지 않음 |
| RL/TTRTC/속도·가속도 limiter | 이 경로에는 없음 |

따라서 현재 코드를 논문의 원식을 그대로 실행하는 구현으로 소개하지 않는다. 원형의 곡선 근사·연결 아이디어에 현재 실행기의 공간 정렬, 수용 검사, 유한 종단과 ROS 전송 방식을 결합한 구현이다. 수식의 가중치·허용량은 현재 코드 상수이지 논문에서 증명된 최적값이라고 볼 수 없다.

### 14. 검증과 튜닝 시 확인할 항목

관련 테스트는 `cyclo_brain/policy/common/runtime/tests/test_rail*.py`, SDK의 `tests/test_rail_transport.py`에 있다. 최근접 위상, 그리퍼 독립 시간축, C² 연결, 정상 방향 전환, 추가 후퇴 거부, 지연·종단 유지, epoch 초기화, q/v/a 전송 등을 나눠 확인한다.

성능 비교는 같은 모델·초기 자세·정책 주기·작업으로 진행하고, 다음을 함께 본다.

- 원본 청크 대비 명령 참조와 실측 관절 각각의 jerk RMS/피크. 샘플 간격과 미분/필터 조건을 동일하게 유지한다.
- 청크 수용률, 거부 사유, 끝점 hold 비율, 계획 시간 및 실제 발행 간격.
- TCP 추가 후퇴, 관절 왕복, 실측 추종 오차.
- 집기·놓기 성공률, 그리퍼 타이밍, 작업 완료 시간.

Cubic 근사와 retiming으로 jerk가 낮아져도 동작이 지나치게 느려지거나 필요한 작업 모션이 없어질 수 있다. 공간 정렬이 좋아도 물체 인식 오류나 충돌을 해결하지는 못한다. 현재 검사는 명령 경로의 샘플 기반 수용 검사이며, 로봇 전체 안전 감독기나 장애물 회피의 대체 수단이 아니다.

---

## 사용

Action Canvas와 Mission Canvas의 SendCommand LOAD에서 다음과 같이 설정한다.

| 설정 | 값 |
|---|---|
| model | groot:n17 |
| policy_path | 기존 일반 GR00T 체크포인트 |
| action_request_mode | async |
| vla_rail_enabled | true |

별도 학습 없이 일반 GR00T 추론 출력에 적용한다. RAIL의 기본값은 꺼짐이며 LOAD 이후 RESUME으로 실행한다. Sync에는 적용할 수 없고, 이미 로드한 모델의 설정을 바꾸려면 LOAD를 다시 해야 한다. 원본 정책 청크와 명령 갱신 모두 `inference_hz`(기본 15Hz)를 따른다. RAIL의 `output_hz = inference_hz`이며, 일반 `control_hz` 설정은 RAIL 발행 주기나 기본 재생 속도에 영향을 주지 않는다. RAIL에서는 청크 길이를 강제로 재샘플링하는 `TARGET_CHUNK_SIZE`를 사용할 수 없다.

## 최근접 청크 정렬 v3 (2026-10-21)

현재 실행 경로는 논문의 전역 cubic fitting을 유지하고, 청크 정렬·수용 및 연결 곡선 선택을 개선한 실험 구현이다. 논문의 Eq.10 부호 점수와 Eq.11~12 dual-quintic 원형은 `rail_trajectory.py` 및 독립 수식 테스트에 비교 기준으로 남겨둔다. 실행 시에는 `rail_handoff.select_handoff()`를 사용하며, 논문 원식을 그대로 실행하는 모드라고 해석하면 안 된다.

v2 기록에서는 본체를 느리게 재생하면서 다음 청크는 관측 경과 시간만큼 앞에서 선택하여, 연결부만 빠르게 따라잡는 현상이 확인됐다. v3는 현재 명령에 가까운 새 경로의 위치를 찾아 이 보정 거리를 줄인다. 위치·속도·가속도의 연속성(C²)만으로 역행 방지를 보장할 수 없으므로 실제 연결 곡선 검사도 유지한다.

1. 일반 비동기 추론으로 원본 액션 청크를 요청한다. prefix conditioning이나 재학습은 사용하지 않는다.
2. 그리퍼를 제외한 각 관절에 대해 원본 전체 청크를 하나의 3차 다항식으로 최소제곱 fitting한다. 원본 waypoint를 정확히 통과하는 보간이 아니다. 이번 변경은 fitting 전 prefix crop을 추가하지 않는다.
3. 관측의 최신성과 팔의 진행 위치를 분리한다. `observation_age × playback_rate >= H`인 만료 청크는 거부한다. 유효 청크는 기존 명령의 교체 시점 q/v/a를 기준으로, 선택된 모든 팔 TCP의 제곱 거리 합이 작은 cubic 위의 연속 시점을 찾는다. 아직 실행하지 않은 기존 청크의 마지막 예측점이나 마지막으로 발행했던 과거 샘플을 기준점으로 쓰지 않는다. 최초 시작은 offset 0에서 최신 피드백으로 연결한다.
4. 최근접 시점의 연결을 먼저 평가하고, 남은 구간이 없거나 연결에 실패하면 공간적으로 가까운 제한된 대안을 검사한다. 관측 나이에 해당하는 먼 시점으로 되돌아가는 fallback은 없다. 실제 연결 곡선의 위치·속도 차이, 불필요한 왕복 및 jerk를 평가한다. 로봇 설정의 URDF로 TCP 위치를 계산하여 원본 예측에 없던 추가 후퇴 거리를 검사하고, 관절 경로 검사도 유지한다. 새 예측의 정상적인 방향 전환은 비교 기준에 포함한다.
5. 수용할 후보가 없으면 새 청크를 버리고 기존 참조를 계속 실행하면서 다음 추론을 요청한다. 기존 유효 horizon이 끝나면 종단 정지 후 유지한다. 기존 곡선을 끝없이 외삽하거나 새 청크 전체를 평행 이동하지 않는다.
6. 수용된 연결은 기존 명령의 현재 q/v/a에서 시작한다. 이후 새 cubic 참조를 실행한다. 실측 피드백은 추종 오차 기록에 사용하고 매 청크의 명령 시작점을 실측으로 되돌리지 않는다.

v3는 v2의 시작·끝 q/v/a를 만족하는 단일 quintic 후보를 유지한다. 논문 원형의 dual-quintic은 비교 기준과 기존 API에 남겨둔다. head 연결 시간과 tail 정지 시간은 독립적이다. 말단 검사는 head·body·tail을 포함한다. 기존 명령의 TCP 속도 방향을 기준으로 각 시점까지 누적된 후퇴를 원본 참조와 비교한다. 추가 후퇴 허용량은 5mm와 해당 시점까지 원본 말단 경로 길이의 2%를 합한 값이다. 2026-10-21 사용자 요청으로 고정 허용량을 3mm에서 5mm로 완화했다. 이전 3mm 기준에서 거부된 6개 청크의 동일 경계 재생에서는 3개가 수용으로 바뀌었고, 추가 후퇴 2개와 만료 1개는 계속 거부됐다. 호스트와 서버에서 같은 결과를 확인했으며 실제 실행 결과는 별도 비교한다. 뒤에 있는 의도적 복귀 구간이 앞서 발생한 불필요한 후퇴를 상쇄하지 못하도록 각 prefix에서 비교한다. 선택 후보는 더 촘촘한 샘플로 다시 검사한다. 이 값은 속도·가속도 제한이 아닌 경로 수용 허용량이며, 샘플 기반 검사만으로 실제 로봇의 무역행이나 충돌 회피를 보장하지 않는다.

전역 fitting은 원본 끝점과 일치하거나 입력 위치 범위 안에 머문다는 보장이 없다. 실행 경로는 `enforce_envelope=False`를 사용한다. 이 개선은 청크 수용·연결 선택이며, 절대 속도·가속도 상한이나 출력 clipping을 추가하지 않는다. 실제 작업 의도와 말단 공간의 진행 방향을 완전히 판별하는 기법도 아니다. 관절 공간의 경로 차이와 말단의 추가 후퇴를 함께 평가하며, 실기에서 추종 오차와 작업 성공 여부를 확인해야 한다. TCP 계산에는 설정 YAML의 `urdf_path`를 사용한다. 필요한 로봇 모델을 읽거나 검증하지 못하면 RAIL LOAD가 실패하며 관절 판정만으로 조용히 우회하지 않는다. FFW의 선택된 팔 순서를 유지하고, 그리퍼는 FK 관절에서 제외한다. lift는 상대 이동량 계산에서 상쇄되는 공통 평행 이동이므로 0으로 고정한다.

## 요청 시점과 시간축

추론 요청은 한 번에 하나만 실행한다. 앞선 요청과 응답 처리가 끝나면 다음 요청을 계속 진행한다. 새 청크는 다음 정책 tick에서 수용 여부를 판단하며, 거부해도 다음 추론 요청을 중단하지 않는다. 실패 응답에는 짧은 재시도 대기가 있다.

`inference_hz`는 원본 정책 액션의 시간 간격이며 실제 GPU 추론 요청 빈도와 다르다. 원본 청크가 `N`개이면 정책 horizon은 `H = (N - 1) / inference_hz`초다. 명령 갱신은 `output_hz = inference_hz`이며, 15Hz이면 약 66.7ms마다 발행한다. 새 추론 청크 도착 시에만 발행하는 방식은 아니다. generic `control_hz`에 맞춘 재샘플링은 하지 않는다.

설정 재생 배율은 `α = output_hz / interpolation_hz`다. Main 환경변수 `VLA_RAIL_INTERPOLATION_HZ`가 없으면 `α = 1`이다. `nominal_offset = observation_age × α`는 만료 판단과 그리퍼 시간 기준이다. v3 팔 offset의 하한은 아니다. 설정값은 양수이며 유한해야 한다. `control_hz`는 발행 주기나 이 계산에 영향을 주지 않는다.

v2에서 도입한 다음 응답까지의 실행 시간 확보를 유지한다.

- 유효 응답의 왕복 지연으로 `response_budget = max(이번 왕복 지연, 이전 budget × 0.9)`를 갱신한다. 만료된 워밍업 응답은 budget에 포함하지 않는다.
- 최소 실행 시간은 `max(2 × 기본 blend, response_budget + 기본 blend + 2 / output_hz)`다.
- 선택된 offset 이후의 재생 배율은 `β = min(α, (H - offset) / 최소 실행 시간)`이다. 발행 주기는 그대로 `output_hz = inference_hz`다.
- 실행 중 정책 시각은 `offset + β × 경과 시간`이고, 새 참조의 속도·가속도에는 `β`, `β²`를 적용한다. 연결 시작 q/v/a는 기존 실행 경로의 값을 그대로 사용한다.
- 만료 판단에는 설정 배율 `α`를 사용한다. 유효 청크의 공간 정렬 시점은 nominal offset보다 앞일 수 있다. 그 지점에 해당하는 과거 명령을 그대로 재생하는 대신, 현재 q/v/a에서 새 미래 연결 곡선을 만든다.
- 예측 끝점을 넘어서 cubic을 외삽하지 않는다. 마지막 예측 위치까지 실행한 뒤 tail로 정지하고 유지한다. `observation_age × α >= H`인 만료 청크는 계속 거부한다.

예를 들어 1초 청크가 관측 후 약 0.665초에 도착해도 실제 팔이 새 예측의 0.1초 위치와 가까우면 그 부근에서 합류할 수 있다. 남은 팔 경로가 길어져 불필요한 따라잡기와 과도한 감속을 함께 줄일 수 있다. 경로를 수용한다는 보장은 없으며, 큰 추가 후퇴를 만드는 후보는 여전히 거부한다.

그리퍼는 공간 거리와 별도 시간축을 사용한다. 시작 시각은 bootstrap에서 0, 이후 `nominal_offset`이고 배율은 `min(α, (H - gripper_offset) / 최소 실행 시간)`이다. 이는 최근접 팔 지점을 선택했다는 이유만으로 그리퍼가 과거 열림값을 재생하거나 마지막 닫힘값으로 건너뛰는 것을 피하기 위한 관측 시간 기준이다. 그리퍼 값 자체를 단조 증가/감소로 제한하지 않으므로 의도된 열기·닫기를 허용한다. 새 예측마다 그리퍼 의도가 달라질 수 있고 팔과 그리퍼의 시간축이 달라지므로 실제 집기·놓기 시점도 실기에서 검증한다.

최초 bootstrap은 피드백에서 offset 0으로 연결한다. 이후 head와 tail은 독립된 시간으로 계산한다. 짧은 잔여 시간 안에 큰 보정을 압축하거나 위치·속도·가속도를 clipping하는 방식은 사용하지 않는다.

## 추종 오차와 거부 사유

새 청크 도착 시 최신 관절 피드백을 읽어 기존 명령 참조와의 최대/RMS 오차를 기록한다. 참조 속도 방향에 투영한 `projected_reference_lead_s`는 국소적인 상태 차이 추정치이며 실제 통신 지연을 측정한 값이 아니다. 움직임이 없는 참조에서는 이 값을 생략한다.

피드백이 초기화에 필요한 최초 시작에서는 유효한 값이 필수다. 실행 중 진단용 피드백을 얻지 못하면 `rail_tracking_unavailable`에 사유를 기록한다. 피드백으로 명령 위치를 리셋하지 않는다. 실행 배율 조정은 위의 응답 지연 budget에 의하며 피드백 지연 추정값을 그대로 적용하지 않는다. `rail_handoff_decision`에는 최근접 시점·거리·후보 목록·선택 순위·대안 사용 여부를 기록하고, `rail_chunk_installed`에는 `nearest_tcp_handoff_v3` 및 팔/그리퍼 각각의 offset·배율을 기록한다.

청크 거부 시 `rail_chunk_rejected`에 이유, 연속 거부 횟수, 기존 참조의 남은 시간과 fallback을 기록한다. 반복 거부로 종단 유지가 많아지는지도 테스트에서 확인해야 한다. 진행 방향 제약을 무조건 완화해 오래된 예측을 강제로 실행하는 우회는 사용하지 않는다.

## 센서 시각과 정렬

`RobotClient.get_observation_snapshot()`은 선택된 카메라·관절 데이터와 해당 콜백의 monotonic 수신 시각을 같은 잠금에서 복사한다. GR00T는 모델이 사용하는 센서만 선택하며 카메라 별칭도 반영한다.

- `sensor_snapshot_monotonic_s`: 선택 관절 중 가장 오래된 수신 시각. 관절이 없으면 선택 이미지 기준.
- `snapshot_read_monotonic_s`: 데이터 복사 시각.
- `sensor_received_monotonic_s`, `sensor_age_s`: 센서별 수신 시각과 나이.
- `sensor_skew_s`: 선택 센서 수신 시각의 최대 차이.
- `timestamp_source=local_monotonic_receive`: 하드웨어 취득 시각이 아니라 로컬 수신 시각이라는 표시.

Main과 Engine의 Linux boot `clock_id`가 일치하면 요청 시작보다 앞선 센서 수신 시각도 정렬에 사용한다. 시계가 다르거나 시각이 유효하지 않으면 요청 시작 시각으로 대체하고 사유를 기록한다. 구형 SDK는 명시적인 `cache_read_fallback`을 사용한다. 새 snapshot API가 실패한 경우에는 이전 캐시 경로로 우회하지 않는다.

`GROOT_SENSOR_MAX_AGE_S`의 기본값은 1초이며 이를 넘긴 센서는 추론 전에 거부한다. `GROOT_SENSOR_MAX_SKEW_S`는 지정하면 시각 차이를 검사하고, 미지정이면 기록만 한다. 두 설정은 Engine 시작 환경변수다. 수신 시각이 노출·취득 시각과 같다는 보장은 없으며, atomic snapshot이 센서 자체를 동기화하지는 않는다.

기록된 관측 나이는 실제 시간이며 정렬 기준은 여기에 재생 배율 `α`를 곱한 정책 시간이다. 수신 시각을 확보해도 서로 다른 시점에 촬영·측정된 센서 값이 같은 시점의 관측으로 바뀌지는 않는다. 센서 age/skew와 모델 입력 시점의 일관성은 별도로 확인해야 한다.

## 로봇 발행과 preview

RAIL은 `publish_action_trajectory()`로 현재부터 3개 출력 주기 뒤까지 위치·속도·가속도와 `time_from_start`를 발행한다. 기본 15Hz에서는 200ms 구간이다. 메시지의 기본 지점은 0, 66.7, 133.3, 200ms이며 다음 발행까지의 구간을 덮는다. 원본 청크 fitting과 연결 곡선은 유지하고 메시지 발행 주기만 정책 주기로 맞춘다. head/tail 경계와 각 dual-quintic의 중간 경계를 추가 지점으로 포함한다. 전역 cubic에는 원본 waypoint마다 별도의 보간 knot가 없다. 시간은 ROS Duration의 나노초 해상도로 변환하며, 매우 가까운 경계는 합쳐 극히 짧은 구간의 수치적 가속도 증폭을 방지한다. 한 그룹이라도 입력이 유효하지 않으면 발행을 시작하지 않는다.

ROS JointTrajectoryController의 `splines` 방식은 위치·속도·가속도가 있으면 quintic 보간을 사용한다. 실제 하위 제어기가 이 설정을 지원해야 한다. 첫 지점이 `t=0`이면 기존 상태에서 첫 지점까지의 보간은 없으며, Jazzy 문서의 궤적 교체는 기존 궤적을 버리는 방식이다. [ROS 제어기 문서](https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/trajectory.html)

메시지 header는 0으로 수신 즉시 실행하는 상대 시간 방식이다. 동기화된 절대 실행 시각이나 실제 하드웨어의 C2 추종을 보장하지 않는다. 발행 중단이 미래 구간보다 길면 창 끝에 도달할 수 있으므로 실기에서 발행 간격과 추종 오차를 확인해야 한다.

그리퍼는 현재 시점의 원본 이산 목표를 창 전체에 동일하게 넣고 속도·가속도는 0으로 보낸다. 미래 grasp를 미리 보간하지 않고 다음 제어 주기에 목표를 갱신한다. 최초 발행은 최신 피드백에서 시작한다. generic Sync/Async의 `publish_action()`은 기존 경로를 유지한다.

preview는 단일 작업자가 최신 대기 값 하나만 처리한다. 느린 시각화 발행이 다음 제어 주기를 막거나 오래된 pose가 누적되지 않는다. STOP·CLEAR·지시/모드 변경 시 대기 preview도 비운다. 이미 시작된 preview 콜백은 끝날 수 있지만 로봇 명령은 발행하지 않는다.

## 상태 전환과 기록

속도·가속도 제한 기능과 관련 설정은 제거했다. `rail_output_guard.py`는 최신 초기 피드백, 유효한 액션 차원·수치, 양수이며 `max(0.1초, 3 / output_hz)` 이하인 명령 간격만 확인한다. 기본 15Hz에서는 0.2초이며, 낮은 정책 주기의 정상 발행이 고정 0.1초 검사에 걸리지 않도록 출력 주기에 맞춘다. RAIL에서는 별도 1점 initial_pose_sync 대신 피드백에서 시작하는 연결 궤적을 사용한다.

STOP·CLEAR·지시 변경은 실행 세대를 갱신하여 이전 응답을 폐기한다. STOP의 피드백 확인 hold가 이미 보낸 미래 구간을 대체한다. 발행 오류나 출력 검사 실패 시 기존 정지 경로로 들어간다. BT JointControl과 다른 추론 모드의 출력 경로는 유지한다.

`INFERENCE_TRACE_PATH`는 일반 추론 로그 경로다. `VLA_RAIL_CAPTURE_ACTIONS=1`이면 원본 청크와 액션도 기록한다. `/workspace/audit/vla_rail/record.enabled`가 있으면 GR00T Main 시작 시 해당 경로 아래 기록을 활성화한다.

| 이벤트 | 내용 |
|---|---|
| `rail_kinematics_configured` | URDF 경로·해시·TCP 링크·좌표계 |
| `rail_raw_chunk` | 원본 청크·정책 간격·센서 timing |
| `rail_transition` | 이전 q/v/a·연결 기준·head/tail 시간·head 종류·판단 상세 |
| `rail_handoff_decision` | 수용 여부·이유·후보 평가값·계획 계산 시간·응답 budget·최소 실행 시간 |
| `rail_chunk_rejected` | 거부 이유·연속 거부 횟수·남은 기존 참조 시간 |
| `rail_tracking`, `rail_tracking_unavailable` | 참조와 피드백 오차 또는 피드백 진단 실패 |
| `rail_chunk_installed` | offset·실행/정책 horizon·재생 배율·관측 나이·정렬 기준·output_hz |
| `rail_sample`, `rail_output_sample` | 참조 q/v/a와 검사 후 액션 |
| `command_published` | sample-to-publish 지연·실제 발행 간격·미래 구간 길이·output_hz |

## 기존 검증·배포 이력 (이번 문서 작업의 재측정 결과가 아님)

최근접 정렬 v3는 **2026-10-21 13:57 KST**에 `groot_server` Main만 재시작해 반영했다. Main PID는 **10166**, Engine PID는 기존 **110**이며 서비스 응답·Engine ping·테스트한 코드 해시 일치를 확인했다. 공통 런타임 **365개 테스트와 16개 subtest**가 통과했다. `output_hz = inference_hz`는 유지하며 2초 액션 간격 설정은 사용하지 않는다.

12:26 실행의 원본 청크 53개를 실제 production 모듈로 재생했다. 기록된 이전 상태를 각각 고정한 비교와 변경된 참조를 다음 연결에 사용하는 순차 재생 모두 **50개 수용, 3개 거부**였다. 순차 재생의 끝점 대기는 **4.75초(12.06%) → 1.56초(3.95%)**, 연결 head의 TCP 최대 속도는 **0.494 → 0.216m/s**, 95백분위는 **0.298 → 0.157m/s**였다. 모든 청크가 느려진 것은 아니며 달라진 연결 상태에서 속도가 증가하는 사례도 있다. 이 결과는 기록된 입력을 고정한 오프라인 비교이며 실제 작업 성공률이나 폐루프 움직임 개선을 보장하지 않는다.

서버 NumPy 1.26.4에서도 53개 × 두 재생 방식의 결정이 호스트와 일치했고 q/v/a 비교의 최대 수치 차이는 `8.51e-7`이었다. 수용한 곡선 전체의 촘촘한 TCP 검사와 C² 경계 검증을 통과했으며 그리퍼 목표는 별도 nominal clock과 일치했다. 서버 순차 재생의 계획 시간은 P95 **36.9ms**, 최대 **40.0ms**였다. 이는 추론과 동시에 실행한 실측 지연이 아니다. 새 trace는 `trace_20261021T045710_10166.jsonl`, 테스트·비교 그래프·서버 결과·배포 기록은 `docker/workspace/audit/vla_rail/nearest_handoff_deploy_20261021/`에 보관한다. BT는 자동 실행하지 않았으며 재시작 후 LOAD부터 진행한다.

최근접 정렬은 연결 지점 선택을 개선한다. 메시지 수신 시각을 시간 원점으로 삼는 기존 제어기 창 교체와 발행 지터는 별도 개선 대상으로 남아 있다.

v2는 2026-10-21 12:06 KST에 서버 반영을 확인했다. 공통 런타임 **345개 테스트와 16개 subtest**, SDK **57개 테스트와 11개 subtest**가 통과했다. 서버 NumPy 1.26.4에서 13개 기록 사례를 3회씩 계산했고, 양팔 후보 계산 최대 약 **32.6ms**, 단팔 약 **21.2ms**였다. 이는 기록 데이터를 이용한 오프라인 측정이며 실제 추론과 동시 실행할 때의 최대 지연을 보장하는 값은 아니다. 호스트와 서버의 수용 결정은 같고, 예제 출력 위치 차이는 최대 `4.44e-15rad`였다.

기록된 관측을 고정한 순차 재생 비교는 다음과 같다. 변경된 로봇 움직임에 따른 새 관측은 생성하지 않으므로 실제 작업 성공률을 뜻하지 않는다.

| 기록 | v1 수용 | v2 수용 | v1 끝점 대기 | v2 끝점 대기 |
|---|---:|---:|---:|---:|
| 11:21 단팔, 145개 | 92 | 103 | 10.25초 (17.6%) | 4.62초 (7.9%) |
| 11:42 양팔, 60개 | 8 | 51 | 35.59초 (90.2%) | 3.06초 (7.8%) |

기존에 약 23mm·51mm·190mm의 큰 후퇴를 만들던 사례는 새 검사에서도 거부했다. 전체 재생·성능·배포 근거는 `docker/workspace/audit/vla_rail/handoff_v2_20261021/`에 보관한다. 최종 Main PID는 410, Engine PID는 110이며 Main/Engine 서비스 응답과 코드 해시를 확인했다. 재부팅으로 중지됐던 GR00T 컨테이너를 기동한 뒤 최종 Main 재시작을 수행했다. BT와 로봇 bringup은 시작하지 않았다. 실제 로봇 검증은 사용자가 bringup 후 LOAD부터 진행한다.

v1 적용 시 공통 런타임 **297개 테스트와 16개 subtest**가 통과했다. 서버 NumPy 1.26.4에서도 전진·뒤처진 경로 거부·정상 방향 전환·q/v/a 전송 재구성을 확인했다. v1 Main 배포는 11:16:43 KST였으며 당시 Engine 프로세스는 유지했다. 해당 배포 기록은 `handoff_20261021/deployment.json`에 보관한다. 이후 사용자가 수행한 실행을 분석한 결과, 11:21 단팔 실행에서는 거부 뒤 종단 대기가 반복됐고, 11:42 양팔 실행에서는 60개 중 52개가 시간 부족으로 거부되어 명령 참조의 약 90%가 끝점 대기였다. 이 결과가 v2의 말단 평가·head/tail 분리·실행 시간 조정 근거다.

논문 수식 검증은 비교 기준으로 유지한다. 개선 경로에는 만료된 관측 거부, 뒤처진 경로의 추가 후퇴 거부, 정상적인 전진·방향 전환 수용, 전체 연결 평가, C² 경계, 거부 후 기존 참조·종단 유지, 다음 요청 재개, STOP·세대 초기화 검증을 추가한다. 기존 5/15/30Hz 발행과 controller 방식의 q/v/a 재보간 검증도 유지한다.

기록된 15Hz 실행의 원본 청크를 사용한 오프라인 비교와 배포 기록은 `docker/workspace/audit/vla_rail/handoff_20261021/`에 보관한다. 과거 관측을 고정한 재생 비교는 로직 검증이며, 변경된 명령에 따라 새로 관측·추론하는 실제 closed-loop 결과를 예측하는 실험이 아니다.

실제 로봇 테스트는 동일한 체크포인트·초기 자세·정책 주기에서 수행한다. 청크 경계 후퇴 거리·추가 방향 반전 횟수·추종 오차와 함께 수용/거부 비율·종단 유지 시간·작업 성공률·완료 시간을 비교한다. RMS 하나가 줄었다는 이유만으로 개선을 판단하지 않는다. 변경 후 BT는 자동 시작하지 않는다.
