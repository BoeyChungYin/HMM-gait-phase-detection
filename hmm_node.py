#!/usr/bin/env python3

import rospy
from hmm_func import hmm
from demo_pkg.msg import rpi_msg
from std_msgs.msg import Int8


def hmm_callback(msg):

    hmm_model.readFromRPi(msg)

if __name__ == "__main__":

    rospy.init_node("hmm_node")
    rospy.loginfo("hmm node started")

    hmm_model = hmm()

    sub = rospy.Subscriber("rpi_state", rpi_msg, callback=hmm_callback)
    pub = rospy.Publisher("hmm_result", Int8, queue_size=10)

    # Set the publishing rate (in Hz)
    rate = rospy.Rate(400)

    while not rospy.is_shutdown():

        prediction = hmm_model.predict()
        pub.publish(prediction)
        rospy.loginfo(f"Predicted State: {prediction}\n")

        # Sleep to control the publishing rate
        rate.sleep()

    rospy.spin()