import tensorflow as tf
import os

tf.flags.DEFINE_string("game", None, "game environment. Ex: Humanoid-v1, OffRoadNav-v0")
tf.flags.DEFINE_string("base-dir", "/exp", "Directory to write summaries and models to.")
tf.flags.DEFINE_string("exp", None, "Optional experiment tag")

def parse_flags():
    FLAGS = tf.flags.FLAGS
    base_dir = os.getcwd()
    FLAGS.exp_dir = base_dir + "{}/{}{}".format(
        FLAGS.base_dir, FLAGS.game, "-" + FLAGS.exp if FLAGS.exp is not None else ""
    )
    # print(FLAGS.exp_dir)
    FLAGS.log_dir = FLAGS.exp_dir + "/log"
    FLAGS.save_path = FLAGS.exp_dir + "/model"
    print(FLAGS.save_path)
    return FLAGS