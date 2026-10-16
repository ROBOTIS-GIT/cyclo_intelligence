// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Seongwoo Kim

import { render, screen } from "@testing-library/react";
import TrainingBalance from "./TrainingBalance";
import { collectionRequest } from "../../../utils/collectionApi";
jest.mock("../../../utils/collectionApi");

test("shows retained supervision labels alongside balanced selection counts", async () => {
 collectionRequest.mockResolvedValue({before:{decisions:{start:2,wait:8,pause:3,continue:4,cancel:2}},
  after:{total:13,decisions:{start:2,wait:2,pause:3,continue:4,cancel:2},missions:{}},supervision_retained:9});
 render(<TrainingBalance mapName="Home" episodeIds={["ep"]} count={19} enabled onChange={jest.fn()} onReady={jest.fn()} />);
 expect(await screen.findByText("Training: 2 start · 2 wait · 3 pause · 4 continue · 2 cancel")).toBeInTheDocument();
 expect(screen.getByText("Available: 2 start · 8 wait · 3 pause · 4 continue · 2 cancel")).toBeInTheDocument();
});


test('shows all three task distributions including retained unknown and minority decisions',async()=>{
 const tasks={mission_selection:{decisions:{start:3,wait:3,unknown:1}},patrol_supervision:{decisions:{continue:2,pause:2}},mission_supervision:{decisions:{continue:4,pause:4,cancel:1}}};
 collectionRequest.mockResolvedValue({before:{tasks},after:{total:20,tasks},excluded:4});
 render(<TrainingBalance mapName="Home" episodeIds={["ep"]} count={24} enabled onChange={jest.fn()} onReady={jest.fn()}/>);
 await screen.findByText('Observation');
 expect(screen.getByText('Patrol')).toBeInTheDocument();
 expect(screen.getByText('Mission')).toBeInTheDocument();
 expect(screen.getByText('Training: 3 start · 3 wait · 1 unknown')).toBeInTheDocument();
 expect(screen.getByText('Training: 4 continue · 4 pause · 1 cancel')).toBeInTheDocument();
});
