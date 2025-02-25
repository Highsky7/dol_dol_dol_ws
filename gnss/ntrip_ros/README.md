# ntrip_ros
NTRIP client, imports RTCM streams to ROS topic

This was forked from github.com/tilk/ntrip_ros

The CORS correction server that I am using does not have the /n/r characters. So I parsed out individual messages and published each one on the /rtcm ROS topic.
It would crash with IncompleteRead error. I added patch at top of file.
But the connection had closed and it would crash again. I ended up detecting zero length data and closing and reopening the data stream.
It continues on without a glitch.

You can generate the require $GPGGA message at this site. https://www.nmeagen.org/ Set a point near where you want to run and click "Generate NMEA file". Cut and paste the $GPGGA message into the launch file.

I intend to use it with https://github.com/ros-agriculture/ublox_f9p

It may also require this package: https://github.com/tilk/rtcm_msgs

A similar NTRIP client (may be better than mine) is here: https://github.com/dayjaby/ntrip_ros
----------------------------------------------------------------------------------------
대승이에게
너가 가지고 있는 모듈은 SparkFun사의 ZED-F9R임
혹시 모듈에 대한 궁금증이 있다면 ZED-F9R HookUp Guide 찾아보셈 -> 각 LED 뭔지도 한 번 찾아봐봐
PPS는 안테나 수신 잘되면 반짝거릴거임.
PWR는 전원 공급 상태이고.
F9P 같은 경우에는 RTK가 기본적으로 HIGH, RTK Float이면 Blink, RTK Fix면 LOW야. F9R의 경우에는 다를 수 있으므로 찾아봐(HookUp Guide gpt에 넣으면 알려줄 수도 있음)

너가 찾아야하는 것은 F9R GPS Driver야. 이것을 다운로드하고 실행해보면 /fix 같은 토픽이 있을거야.
보통 여기에 위도(latitude), 경도(longitude) 정보 등이 있을거야.
Driver만 실행 시에는 RTK가 안된 값이므로 ㅈㄴ게 튈거야.
혹시나 covariance(크기 9의 배열 -> 아마 0,3,6번 index 값들만 나올거임(나머지 0) -> 각각 x,y,z의 1m^2 당 분산이고, 오차는 예를 들어 sqrt(covariance[0]) 미터임) -> gpt나 인터넷 더 찾아보셈

아무튼 Driver를 실행하고 rostopic 확인하고 위도/경도 확인해봐. status도.

이후에 위도 경도가 확인되면 내가 git에 올린 ntrip 정보를 줘야해.

roslaunch ntrip_ros ntrip_ros.launch
를 실행하면 막 무슨 배열들이 뜰거야. 그러면 잘된거야.

둘 다 포트 번호 맞아야 실행되고 
ls /dev/ttyACM 하고 tab 눌러보면 지금 연결된 포트 번호 나옴
그거 맞게 잘 확인하고!!

화이팅이다.

현우가
