# bearing-fault-diagnosis
deep learning methods for bearing fault diagnosis.
## Data
bearing fault datasets are arranged in Dataset Object and packaged in *.pkl files.<br>
Download in this link:<br>

Detail description of these datset is below<br>
CWRU:http://csegroups.case.edu/bearingdatacenter/home<br>
GERMAN:https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/<br>
ours:<br>
8 bearings are tested (2 healthy, 3 inner race fault and 3 outer race fault)<br>
bearing code | K001 | K002 | KI01 | KI02 | KI03 | KA01 | KA02 | KA03<br>
fault place  |  --- |  --- | inner| inner| inner| outer| outer| outer<br>
methods      |  --- |  --- |EMD|drilling|natural|EMD|drilling|natural<br>
## Project 1. domain adaption with adversarial training
The whole network is devided into 3 parts: feature extractor, domain predictor and classifier.
