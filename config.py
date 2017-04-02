import tensorflow as tf
import os

tf.flags.DEFINE_string("mode", "train", "choices=[train, test]")
tf.flags.DEFINE_string("model", "3dense", "NN model")
tf.flags.DEFINE_string("agent", "dqn", "for example, dqn")
tf.flags.DEFINE_string("game", None, "game environment. Ex: Humanoid-v1, OffRoadNav-v0")
tf.flags.DEFINE_string("base-dir", "exp", "Directory to write summaries and models to.")
tf.flags.DEFINE_integer("save-weight-interval", 10000, "number of steps before saving the model weight")
tf.flags.DEFINE_integer("save-log-interval", 100, "number of episodes before saving the model weight")
tf.flags.DEFINE_integer("max-steps", 5000, "max number of steps for the whole training")
tf.flags.DEFINE_boolean("visualize-train", True, "visualize the training process")


def parse_flags():
    FLAGS = tf.flags.FLAGS
    # FLAGS.exp_dir = os.getcwd() + "/{}/{}/{}/".format(FLAGS.base_dir, FLAGS.game, FLAGS.agent)

    # Keras use relative path to save models
    FLAGS.exp_dir = "{}/{}/{}/".format(FLAGS.base_dir, FLAGS.game, FLAGS.agent)

    FLAGS.log_dir = FLAGS.exp_dir + "/log"
    FLAGS.save_path = FLAGS.exp_dir + "/model"
    # print(FLAGS.base_dir, FLAGS.save_weight_interval)
    return FLAGS