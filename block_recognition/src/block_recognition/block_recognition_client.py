import block_recognition_msgs.srv
import block_recognition_msgs.msg
import typing
import rospy


def wait_for_service(timeout=10):
    # type: () -> bool
    try:
        rospy.wait_for_service('/objrec_node/find_blocks', timeout=timeout)
    except rospy.ROSException as e:
        return False

    return True


def find_blocks():
    # type: () -> typing.List[block_recognition_msgs.msg.DetectedBlock]

    service_proxy = rospy.ServiceProxy('/objrec_node/find_blocks', block_recognition_msgs.srv.FindObjects)
    service_proxy.wait_for_service(timeout=5)
    resp = service_proxy()

    return resp.detected_blocks
