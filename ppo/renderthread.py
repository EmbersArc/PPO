import threading
import time


class RenderThread(threading.Thread):
    def __init__(self, sess, trainer, environment, brain_name, normalize):
        threading.Thread.__init__(self)
        self.sess = sess
        self.env = environment
        self.trainer = trainer
        self.brain_name = brain_name
        self.normalize = normalize
        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())

    def run(self):
        with self.sess.as_default():
            while True:
                with self.pause_cond:
                    while self.paused:
                        self.pause_cond.wait()

                    done = False
                    info = self.env.reset()[self.brain_name]
                    while not done:
                        info = self.trainer.take_action(info, self.env, self.brain_name, 0, self.normalize,
                                                        stochastic=False)
                        done = info.local_done[0]
                time.sleep(0.1)

    def pause(self):
        self.paused = True
        self.pause_cond.acquire()

    def resume(self):
        self.paused = False
        self.pause_cond.notify()
        self.pause_cond.release()
