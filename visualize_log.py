#  Copyright 2020 RangerUFO
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import re
import numpy as np
import matplotlib.pyplot as plt


def visualize(lines):
    regex = re.compile("^\[Epoch ([0-9]+)  Batch ([0-9]+)\]  batch_loss (\S+).*")
    batch_x = []
    batch_loss = []
    for line in lines:
        m = regex.match(line)
        if m:
            batch_x.append((int(m.group(1)), int(m.group(2))))
            batch_loss.append(float(m.group(3)))
    batches = max(batch_x, key=lambda x: x[1])[1]
    batch_x = [epoch + batch / batches for epoch, batch in batch_x]
    regex = re.compile("^\[Epoch ([0-9]+)\]  training_loss (\S+)  validation_score (\S+).*")
    epoch_x = []
    training_loss = []
    validation_score = []
    for line in lines:
        m = regex.match(line)
        if m:
            epoch_x.append(int(m.group(1)))
            training_loss.append(float(m.group(2)))
            validation_score.append(float(m.group(3)))
    plt.subplot(2, 1, 1)
    plt.plot(np.array(batch_x), np.array(batch_loss), label="batch loss")
    plt.grid(True)
    axl = plt.subplot(2, 1, 2)
    axl.plot(np.array(epoch_x), np.array(training_loss))
    axl.set_ylabel("training loss")
    axr = axl.twinx()
    axr.plot(np.array(epoch_x), np.array(validation_score), "r")
    axr.set_ylabel("validation score")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    lines = []
    while True:
        try:
            lines.append(input())
        except EOFError:
            break
    visualize(lines)
