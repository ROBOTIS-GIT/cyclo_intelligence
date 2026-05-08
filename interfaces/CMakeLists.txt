cmake_minimum_required(VERSION 3.16)
project(interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ActionChunk.msg"
  "msg/BrowserItem.msg"
  "msg/DataOperationStatus.msg"
  "msg/DatasetInfo.msg"
  "msg/HFOperationStatus.msg"
  "msg/InferenceStatus.msg"
  "msg/RecordingStatus.msg"
  "msg/TaskInfo.msg"
  "msg/TrainingInfo.msg"
  "msg/TrainingProgress.msg"
  "msg/TrainingStatus.msg"
  "srv/BrowseFile.srv"
  "srv/ControlHfServer.srv"
  "srv/EditDataset.srv"
  "srv/GetConversionStatus.srv"
  "srv/GetDatasetInfo.srv"
  "srv/GetDatasetList.srv"
  "srv/GetHFUser.srv"
  "srv/HFEndpointList.srv"
  "srv/HfOperation.srv"
  "srv/SelectHFEndpoint.srv"
  "srv/GetImageTopicList.srv"
  "srv/GetModelWeightList.srv"
  "srv/GetPolicyList.srv"
  "srv/GetReplayData.srv"
  "srv/GetRobotTypeList.srv"
  "srv/GetSavedPolicyList.srv"
  "srv/GetTrainingInfo.srv"
  "srv/GetTreeList.srv"
  "srv/GetUserList.srv"
  "srv/LoadAndRunTree.srv"
  "srv/RecordingCommand.srv"
  "srv/SendCommand.srv"
  "srv/SendTrainingCommand.srv"
  "srv/SetHFUser.srv"
  "srv/SetRobotType.srv"
  "srv/StartConversion.srv"
  "srv/CheckpointList.srv"
  "srv/ModelList.srv"
  "srv/PolicyList.srv"
  "srv/InferenceCommand.srv"
  "srv/StopTraining.srv"
  "srv/TrainModel.srv"
  "srv/TrainingStatus.srv"
)

install(
  DIRECTORY msg srv
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
