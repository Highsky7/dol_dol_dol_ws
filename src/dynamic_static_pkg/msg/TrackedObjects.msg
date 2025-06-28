# TrackedObjects.msg
# SortRos(TrackerState) 로부터 추출한 트랙 정보 메시지
std_msgs/Header    header
int32[]                 id
geometry_msgs/Point[]   center
float32[]               vx
float32[]               vy