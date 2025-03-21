#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from datetime import datetime

from rtcm_msgs.msg import Message

from base64 import b64encode
from threading import Thread

import http.client as httplib

# IncompleteRead 예외 처리를 위해 patch
def patch_http_response_read(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except httplib.IncompleteRead as e:
            return e.partial
    return inner
httplib.HTTPResponse.read = patch_http_response_read(httplib.HTTPResponse.read)


class NtripConnect(Thread):
    def __init__(self, ntc):
        super(NtripConnect, self).__init__()
        self.ntc = ntc
        self.stop = False

    def run(self):
        # Base64 인코딩 시 Python 3에서는 bytes 처리가 필요
        auth_str = f"{self.ntc.ntrip_user}:{self.ntc.ntrip_pass}"
        auth_b64 = b64encode(auth_str.encode()).decode()

        headers = {
            'Ntrip-Version': 'Ntrip/2.0',
            'User-Agent': 'NTRIP ntrip_ros',
            'Connection': 'close',
            'Authorization': 'Basic ' + auth_b64
        }

        # body 역시 bytes로 전달
        body_data = self.ntc.nmea_gga.encode()

        connection = httplib.HTTPConnection(self.ntc.ntrip_server)
        connection.request('GET',
                           '/' + self.ntc.ntrip_stream,
                           body_data,
                           headers)
        response = connection.getresponse()

        if response.status != 200:
            raise Exception("NTRIP server responded with status code: %d" % response.status)

        buf = b""  # Python 3에서는 이진 데이터 처리를 위해 bytes 사용
        rmsg = Message()
        restart_count = 0

        while not self.stop:
            # RTCM 패킷 분리 로직
            data = response.read(1)
            if len(data) != 0:
                # RTCM 패킷의 시작은 0xD3(십진수 211)
                if data[0] == 211:
                    buf += data
                    # 2바이트 읽어 메시지 길이 추출
                    length_data = response.read(2)
                    buf += length_data
                    cnt = length_data[0] * 256 + length_data[1]

                    # 다음 2바이트에서 메시지 유형 등 정보 확인
                    type_data = response.read(2)
                    buf += type_data
                    typ = (type_data[0] * 256 + type_data[1]) // 16

                    # 메시지 길이(cnt)에 따라 해당 바이너리 데이터를 읽어옴
                    cnt = cnt + 1  # parity(체크섬) 포함
                    for _ in range(cnt):
                        d = response.read(1)
                        buf += d

                    # 메시지 발행
                    # rtcm_msgs/Message에 message 필드가 string이라면 디코딩이 필요할 수도 있음
                    # 여기서는 raw bytes를 그대로 담아 보낸다고 가정
                    rmsg.message = buf
                    rmsg.header.seq += 1
                    rmsg.header.stamp = rospy.get_rostime()
                    self.ntc.pub.publish(rmsg)
                    buf = b""

                    # 디버그 출력
                    print(str(datetime.now()), "length:", cnt, "type:", typ)
                else:
                    # D3 헤더가 아니라면 디버그용 출력
                    print("Non-RTCM data:", data)
            else:
                # 서버로부터 데이터를 못 받으면 재접속
                restart_count += 1
                print("Zero length data, reconnecting:", restart_count)
                connection.close()
                connection = httplib.HTTPConnection(self.ntc.ntrip_server)
                connection.request('GET',
                                   '/' + self.ntc.ntrip_stream,
                                   body_data,
                                   headers)
                response = connection.getresponse()
                if response.status != 200:
                    raise Exception("NTRIP server responded with status code: %d" % response.status)
                buf = b""

        connection.close()


class NtripClient:
    def __init__(self):
        rospy.init_node('ntripclient', anonymous=True)

        self.rtcm_topic = rospy.get_param('~rtcm_topic', 'rtcm')
        self.nmea_topic = rospy.get_param('~nmea_topic', 'nmea')

        self.ntrip_server = rospy.get_param('~ntrip_server')
        self.ntrip_user = rospy.get_param('~ntrip_user')
        self.ntrip_pass = rospy.get_param('~ntrip_pass')
        self.ntrip_stream = rospy.get_param('~ntrip_stream')
        self.nmea_gga = rospy.get_param('~nmea_gga')

        self.pub = rospy.Publisher(self.rtcm_topic, Message, queue_size=10)

        self.connection = None
        self.connection = NtripConnect(self)
        self.connection.start()

    def run(self):
        rospy.spin()
        if self.connection is not None:
            self.connection.stop = True


if __name__ == '__main__':
    c = NtripClient()
    c.run()
