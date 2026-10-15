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
