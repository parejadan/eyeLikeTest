eyeLikeTest
-----------
This repo is the testing sandbox for improving pupil detection used in lifeSaver app.

Updates
-------
- training module added for Monitor class.
- tracking monitor "watch()" added, - not completely functional or tests have not been properly conducted.

Issues
------
- pupil detection algorithm (or code) does not provide distinguishing signals whenever eyes are not closed. This is due it detecting eyelashes as pupils
- statistical model used for monitoring might not be suitable for our intended dataset; 