import cv2

def find_available_cameras(max_index=10):
    available_ports = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_ports.append(i)
            cap.release()
    return available_ports

if __name__ == "__main__":
    ports = find_available_cameras()
    if ports:
        print("사용 가능한 카메라 포트:", ports)
    else:
        print("사용 가능한 카메라가 없습니다.")
