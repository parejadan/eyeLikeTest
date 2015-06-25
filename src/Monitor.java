

import java.util.ArrayList;
import java.util.Date;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.objdetect.CascadeClassifier;

import app.ai.lifesaver.imgproc.FindEyes;
import app.ai.lifesaver.imgproc.Vars;

public class Monitor {

    private static FindEyes fd = new FindEyes();
    private static CascadeClassifier mJavaDetector; //java classifier
    private static float mRelativeFS = 0.2f;
    private static int mAbsoluteFS = 0;
    private static double closedThresh;
    private static double placeVal;

    private static double std, key;
    private static long cloCheckTime;
    private static long magCheckTime;
    boolean cloFlag;
    boolean magFlag;
    
    /**
     * @param mJDetector - cascade classifier for face detection
     * @param ct - percent pupil 
     * @param at - acceptable time eyes can remain closed or not present (milliseconds)
     */
    public Monitor(CascadeClassifier mJDetector, double ct, double pVal, long tl, long trnT) {
    	mJavaDetector = mJDetector;
    	closedThresh = ct;
    	placeVal = pVal;
    	cloCheckTime = tl;
    	magCheckTime = trnT;
    }
	
	/** From a given image, it locates the largest face and identifies the right pupil position 
	 *  relative to the overall image.
	 * @param fram - raw image, make sure to "release" after method call
	 * @return coordinates for the right pupil; (-1,-1) if pupils or face were not found
	 */
	Point getPupilxy(Mat fram) {
		Point pupil = new Point();
		ArrayList<Mat> mv = new ArrayList<Mat>(3);
        MatOfRect faces = new MatOfRect();
        
        Core.split(fram , mv);
        Mat mGray = mv.get(2);
        if (mAbsoluteFS == 0) //if face size is not initialize, estimate a possible minimum size
        	mAbsoluteFS = (int) (mGray.rows() * mRelativeFS);
        
        //locate faces in image
        mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2,
        		new org.opencv.core.Size(mAbsoluteFS, mRelativeFS), new org.opencv.core.Size());
        Rect[] facesArray = faces.toArray();

        if (facesArray.length == 0) { //eyes cannot be located
        	pupil.x = -1; pupil.y = -1;
        	
        } else { //determine driver's face by identifying the largest object
            Rect mxFace = facesArray[0];
            for (Rect tmp : facesArray)
            	if (tmp.height+tmp.width > mxFace.height+mxFace.width)
            		mxFace = tmp;
            
        	pupil = fd.findPupil(mGray, mxFace, false); //locate right eye automatically
    		Core.circle(mGray, pupil, 5, Vars.EYE_CIRCLE_COLR);
    		//System.out.printf("perc: %f\t|thresh: %f\n", FindEyes.cEye[1]/pupil.y, closedThresh);
        	if ( FindEyes.cEye[1]/pupil.y >= closedThresh ) { //check if eyes are closed
        		pupil.x = -1; pupil.y = -1;
        		//System.out.println("closed eyes");
        	} else {
            	//represent pupil coordinate as a ratio relative to eye region
        		pupil.x -= FindEyes.cEye[0]; pupil.x /= FindEyes.eyeRegW; pupil.x *= placeVal;
        		pupil.y -= FindEyes.cEye[1]; pupil.y /= FindEyes.eyeRegH; pupil.y *= placeVal;           	
        	}
        }
		
		//Core.circle(mGray,  pupil,  5,  Vars.EYE_CIRCLE_COLR);
    	Highgui.imwrite("detection.png", mGray);

        for (Mat tmp : mv) tmp.release();
        mv.clear();
        faces.release(); mGray = faces = null;
        
        return pupil;
	}

	int length(Point a, Point b) {
		//double mag = Math.sqrt( Math.pow(a.x-b.x, 2.0) + Math.pow(a.y-b.y, 2.0) );
		//System.out.printf("a.x: %f\t|b.x: %f\t|a.y: %f\t|b.y: %f\t|m: %f\n", a.x, b.x, a.y, b.y, mag);
		return (int) Math.sqrt( Math.pow(a.x-b.x, 2.0) + Math.pow(a.y-b.y, 2.0) );
	}
	
	/** Takes in a list of magnitudes and computes the STD from the key
	 * @param mgs - list of magnitudes recorded for pupil movement
	 * @param placeVal - if dealing with hole numbers, set to 1, if dealing with values between
	 * 	[1,0] set to to the desired decimal place. Ex. .234 for 10ths -> set placeVal to 10 for 23
	 * @return - array where value at index 0 is std, and value at index 1 is key
	 */
	double[] computeKeySTD(ArrayList<Integer> mgs) {
		double[] sigKey = new double[2]; //sigKey[0] = standard derivation; sigKey[1] = Key
		int size = mgs.size();
		/*int[] sortedLST;
		int size = mgs.size(), min = mgs.get(0), max = mgs.get(0);
		for (int num : mgs) {
			if (num > max)
				max = num;
			else if (num < min)
				min = num;
			System.out.printf("%d,", num);
		}
		sortedLST = new int[max-min+1];
		for (int num : mgs) {//perform count sort to distinguish mode (num is treated as a index)
			sortedLST[num-min]++;
			System.out.printf("dex: %d\t|num: %d\t|occurance: %d\n", num-min, num, sortedLST[num-min]);
		}
		//locate mode, mx is the maximum occurrence of a number from the original list
		for (int mx = sortedLST[0], i = 1; i < max-min; i++) {
			if (sortedLST[i] > mx) {
				sigMod[1] = min+i; //set mode
				mx = sortedLST[i];
			}
			System.out.printf("mode: %f\n", sigMod[1]);
		}*/
		
		for (int num : mgs) { sigKey[1] += num;System.out.printf("%d,", num);}
		sigKey[1] /= size;

		//compute sample standard derivation
		for (int  num : mgs)
			sigKey[0] += Math.pow(sigKey[1]-num, 2.0); //compute variance using a "key"

		sigKey[0] /= size-1;
		sigKey[0] = Math.sqrt(sigKey[0]);

		return sigKey;
	}

	/** Checks if the eye movement falls bellow a standard deviation from  regular eye movement
	 * @param mag - magnitude to check on
	 * @return true if it falls bellow it, false otherwise
	 */
	boolean checkMovement(double mag) { return (mag < key-std) ? true : false; }

	void train(VideoCapture cam) {
		ArrayList<Point> tracking = new ArrayList<Point>();
		Mat frame = new Mat();
		Point tmp, tmp2;
		
		//record frames
		long mgSTime = new Date().getTime(), cTime = new Date().getTime();
		while (cTime-mgSTime <= magCheckTime) {
			//System.out.println(cTime-mgSTime);
			cam.read(frame);
			tracking.add(getPupilxy(frame));
			frame.release();
			cTime = new Date().getTime();
		}
		//get baseline Key and std from training data
		ArrayList<Integer> mags = new ArrayList<Integer>();
		double[] sigMod;
		int i, size = tracking.size();
		tmp = tracking.get(size-1);
		for (i = size-2; i >= 0; i--) {
			tmp2 = tracking.get(i);
			//treat closed eye signals as no eye movement
			if (tmp.x == -1)
				mags.add( length(new Point(0, 0), tmp2) );
			else if (tmp2.x == -1)
				mags.add( length(tmp, new Point(0, 0)) );
			else
				mags.add( length(tmp, tmp2) );	
			tmp = tmp2; //rotate coordinate for next computation
		}
		
		sigMod = computeKeySTD(mags);
		std = sigMod[0];
		key = sigMod[1];
		System.out.printf("standard: %f\t|key: %f\n", std, key);

	}
	
	/**
	 * Monitors drivers pupil movement to see if they are acceptably active or not.
	 * cloFlag - true whenever a driver is considered to have eyes closed for an extended period
	 * magFlag - true whenever a drivers eye movement falls bellow an acceptable baseline (obtained from training)
	 * @param cam - pointer to video camera where frames can be extracted from
	 */
	void watch(VideoCapture cam) {
		Records tracking = new Records(magCheckTime);
		Mat frame = new Mat();
		double clCnt;
		int i, size, count;
		Recs tmp, tmp2;
		
		long cTime, clSTime = new Date().getTime(), mgSTime = new Date().getTime();
		while(true) {
			cam.read(frame); //read another frame
			tracking.add(getPupilxy(frame), new Date().getTime()); //processes frame for pupil coordinates; record time        	
			frame.release();
			size = tracking.size();
			cTime = tracking.data.get( size-1 ).getT(); //use logged time of most recent frame added as current

			//CLOSED EYES DETECTION UNDER CONSTRUCTION
			/*if (cTime-clSTime >= cloCheckTime) { //time to check for closed eyes
				System.out.println("checking closed eyes");
				for (i = size-1, count = 0, clCnt = 0; i >= 0; i--) { //we look from the most recent frame to to the oldest
					tmp = tracking.data.get(i);
					System.out.printf("%f\n",tmp.getP().x);
					if (cTime-tmp.getT() <= cloCheckTime) {//check if frame for closed eyes apply time interval to check
						if (tmp.getP().x == -1) {//closed eyes frame detected
							System.out.println("closed eye!");
							clCnt++;
						}
						count++;
					} else {
						break; //frames have either already been checked or too old to mater
					}
				}
				if (clCnt / count >= .6)
					cloFlag = true; //eyes are closed, flag this bitch up
				else
					cloFlag = false; //you got off easy this time
				clSTime = cTime;
			}*/
			
			if (cTime-mgSTime >= magCheckTime) { //time to check eye magnitudes
				ArrayList<Integer> mags = new ArrayList<Integer>();
				double[] sigKey;
				//calculate magnitudes
				tmp = tracking.data.get(size-1);
				for (i = size-2; i >= 0; i--) {
					tmp2 = tracking.data.get(i);
					//System.out.printf("a.x: %f\t|a.y: %f\t|b.x: %f\t|a.y: %f\n", tmp.getP().x, tmp.getP().y, tmp2.getP().x,tmp.getP().y);
					//treat closed eye signals as no eye movement
					if (tmp.getP().x == -1)
						mags.add( length(new Point(0, 0), tmp2.getP()) );
					else if (tmp2.getP().x == -1)
						mags.add( length(tmp.getP(), new Point(0, 0)) );
					else
						mags.add( length(tmp.getP(), tmp2.getP()) );	
					tmp = tmp2; //rotate coordinate for next computation
				}

				sigKey = computeKeySTD(mags); //get std and key from the monitoring time period
				if (checkMovement(sigKey[1]))
					magFlag = true; //eye movement is falling bellow baseline
				else
					magFlag = false; //all checks-out; as you were
				System.out.printf("|checked key: %f\t|%s\n", sigKey[1], (magFlag) ? "bellow baseline!":"");

			}
			
			//System.out.printf("%s\t|%s\n", (magFlag) ? "bellow baseline!":"", (cloFlag) ? "wake up!":"");
		}
	}
}
