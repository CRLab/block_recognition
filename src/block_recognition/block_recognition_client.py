import block_recognition.srv
import block_recognition.msg
import typing
import rospy


class BlockRecognitionClient():
    def __init__(self):
        pass

    def find_blocks(self):
        # type: () -> typing.List[block_recognition.msg.DetectedBlock]

        service_proxy = rospy.ServiceProxy('/objrec_node/find_blocks', block_recognition.srv.FindObjects)
        service_proxy.wait_for_service(timeout=5)
        resp = service_proxy()

        return resp.detected_blocks
