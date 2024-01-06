import random
import pickle
import numpy as np
from jiwer import wer


questions = np.load('SP-train.npy', allow_pickle=True)

with open('unified_qa_zero_shot.pickle', 'rb') as handle:
    uqa = pickle.load(handle)

all_correct = 0
pure_correct = 0
n_pure = 0
for question in questions:
    assert question['id'] in uqa
    answer = str(question['answer']).lower().strip()
    pred = uqa[question['id']]['pred']
    if answer == 'none of above.': continue
    n_pure += 1
    if answer == pred:
        pure_correct += 1
        # print(f"{answer} ==> {pred}")
    else:
        d1 = str(question['distractor1']).lower().strip()
        d2 = str(question['distractor2']).lower().strip()
        print(f"{answer} \t {pred}")

print(pure_correct / n_pure)
print(pure_correct)
print(n_pure)

# i = random.randint(0, (len(x) // 3) - 1)
# while x[i*3]['answer'] == 'None of above.':
#     i = random.randint(0, (len(x) // 3) - 1)

# print(f'Question: {x[i*3]["question"]}\n')
# print(f'Answer: {x[i*3]["answer"]}')
# print(f'Distractor1: {x[i*3]["distractor1"]}')
# print(f'Distractor2: {x[i*3]["distractor2"]}')


logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
  def on_validation_end(self, trainer, pl_module):
    logger.info("***** Validation results *****")
    if pl_module.is_logger():
      metrics = trainer.callback_metrics
      # Log results
      for key in sorted(metrics):
        if key not in ["log", "progress_bar"]:
          logger.info("{} = {}\n".format(key, str(metrics[key])))

  def on_test_end(self, trainer, pl_module):
    logger.info("***** Test results *****")

    if pl_module.is_logger():
      metrics = trainer.callback_metrics

      # Log and save results to file
      output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
      with open(output_test_results_file, "w") as writer:
        for key in sorted(metrics):
          if key not in ["log", "progress_bar"]:
            logger.info("{} = {}\n".format(key, str(metrics[key])))
            writer.write("{} = {}\n".format(key, str(metrics[key])))