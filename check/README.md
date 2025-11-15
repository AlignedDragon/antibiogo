This folder contains codes used to evaluate yolo+xyr model.

## Results:
#### Graphic:
Results are reported based on mm error. If radius is smaller, error is negative. And vice versa.

Improc failed to detect 6% of inhibition zones at all.
Dotted lines coincide with -1mm and 1mm
* Treating improc's zone misses as errors: (Miss)
![Screenshot from 2024-12-04 22-35-55](https://github.com/user-attachments/assets/f03a27cc-fdef-4aaa-bfcf-ed18a8083221)
* NOT treating improc's zone misses as errors: (NoMiss)
![Screenshot from 2024-12-04 22-36-55](https://github.com/user-attachments/assets/88ffe64e-a63c-4cdd-95b4-b6096e4e1050)

#### Written:
In computing errors approximation of (0.49~0 and 0.5~1) was used:

##### Error of more than 1mm:  
xyr - 15.54%, improcNoMiss - 18.35%, improcMiss - 23.70%

##### Error of more than 3mm: 
xyr - 2.59%, improcNoMiss - 9.06%, improcMiss - 15.01%

##### Error of more than 1mm but less than 3mm: (Assumption: more than 3mm is detectable even by lazy user)
xyr - 12.95%, improcNoMiss - 8.29%, improcMiss - 8.68%

#### Breakdown:
Based on rough subjective estimation. 'xyr' models mistakes are due to:
 1) 60% edge-case inconsistency. That's while labelling edge-case ihz radii, there was no consistent rule.
 2) 35% hard for human eye.
 3) 5% neither 1st not 2nd, but inherent to the model.
