

import java.util.ArrayList;
import java.util.Date;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.highgui.VideoCapture;
import org.opencv.objdetect.CascadeClassifier;

import app.ai.lifesaver.imgproc.FindEyes;
import app.ai.lifesaver.imgproc.Vars;

public class Monitor {

    private static FindEyes fd = new FindEyes();
    private static CascadeClassifier mJavaDetector; //java classifier
    private static float mRelativeFS = 0.2f;
    private static int mAbsoluteFS = 0;
    private static int closedThresh;

    private static double std, mode;
    private static long clCheckTime;
    private static long mgCheckTime;
    boolean clFlag;
    boolean mgFlag;
    
    /**
     * @param mJDetector - cascade classifier for face detection
     * @param ct - pixel threshold that determines when a pupil is closed or not;
     * 	- value bellow 127 is too sensitive, above 140 is to dull
     * @param at - acceptable time eyes can remain closed or not present (milliseconds)
     */
    public Monitor(CascadeClassifier mJDetector, int ct, long tl) {
    	mJavaDetector = mJDetector;
    	closedThresh = ct;
    	clCheckTime = tl;
    	mgCheckTime = 60000l;
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
    		Core.circle(mGray,  pupil,  5,  Vars.EYE_CIRCLE_COLR);
    		
        	if ( mGray.get((int) pupil.y, (int) pupil.x)[0] > closedThresh ) { //check if eyes are closed
        		pupil.x = -1; pupil.y = -1;
        	} else {
            	//represent pupil coordinate as a ratio relative to eye region
            	pupil.x /= FindEyes.eyeRegW;
            	pupil.y /= FindEyes.eyeRegH;        		
        	}
        }
		
        for (Mat tmp : mv) tmp.release();
        mv.clear();
        faces.release(); mGray = faces = null;
        
        return pupil;
	}

	double length(Point a, Point b) { return Math.sqrt((a.x - b.x)*(a.x - b.x) + (a.y + b.y)*(a.y + b.y)); }
	
	/** Takes in a list of magnitudes and computes the STD from the mode instead of mean
	 * @param mgs - list of magnitudes recorded for pupil movement
	 * @param placeVal - if dealing with hole numbers, set to 1, if dealing with values between
	 * 	[1,0] set to to the desired decimal place. Ex. .234 for 10ths -> set placeVal to 10 for 23
	 */
	void computeModeSTD(ArrayList<Double> mgs, double placeVal) {
		int[] sortedLST;
		
		//locate min max to then perform a count sort
		int max = (int) Math.ceil(mgs.get(0) * placeVal);
		int min = max;
		for (double num : mgs) {
			if (num > max) max = (int) Math.ceil(num);
			else if (num < min) min = (int) Math.ceil(num);
		}
		sortedLST = new int[max-min];
		for (double num : mgs) sortedLST[(int)num-min]++; //perform count sort to distinguish mode
		//locate mode, mxO is the maximum occurrence of a number from the original list
		for (int mxO = 0, i = 0; i < max-min; i++)
			if (sortedLST[i] > mxO)  {
				mode = min+i;
				mxO = sortedLST[i];
			}
		//compute standard derivation
		double var = 0;
		for (double num : mgs) var += Math.pow(mode-num, 2.0); //variance on mode instead of mean
		var /= mgs.size();
		std = Math.sqrt(var);
	}

	/** Checks if the eye movement falls bellow a standard deviation from  regular eye movement
	 * @param mag - magnitude to check on
	 * @return true if it falls bellow it, false otherwise
	 */
	///boolean checkMovement(double mag) { return (mag < mode-std) ? true : false; }

	void watch(VideoCapture cam) {
		Records tracking = new Records(mgCheckTime);
		Mat frame = new Mat();
		double clCnt = 0;
		int i, size, count;
		Recs tmp, tmp2;
		
		long cTime, clSTime = new Date().getTime(), mgSTime = new Date().getTime();
		while(true) {
			cam.read(frame); //read another frame
			tracking.add(getPupilxy(frame), new Date().getTime()); //processes frame for pupil coordinates; record time
			frame.release();
			size = tracking.size();
			cTime = tracking.data.get( size-1 ).getT(); //use logged time of most recent frame added as current

			if (cTime-clSTime >= clCheckTime) { //time to check for closed eyes

				for (i = size-1, count = 0; i >= 0; i--) { //we look from the most recent frame to to the oldest
					tmp = tracking.data.get(i);
					
					if (cTime-tmp.getT() <= clCheckTime) {//check if frame for closed eyes apply time interval to check
						if (tmp.getP().x == -1) //closed eyes frame detected
							clCnt++;
						count++;
					} else {
						break; //frames have either already been checked or too old to mater
					}
				}
				if (clCnt / count >= .6)
					clFlag = true; //eyes are closed, flag this bitch up
				else
					clFlag = false; //you got off easy this time
				clSTime = cTime;
			}
			
			if (cTime-mgSTime >= mgCheckTime) { //time to check eye magnitudes
				ArrayList<Double> mags = new ArrayList<Double>();
				
				//calculate magnitudes
				tmp = tracking.data.get(size-1);
				for (i = size-2; i >= 0; i--) {
					tmp2 = tracking.data.get(i);
					//treat closed eye signals as no eye movement
					if (tmp.getP().x == -1)
						mags.add( length(new Point(0, 0), tmp2.getP()) );
					else if (tmp2.getP().x == -1)
						mags.add( length(tmp.getP(), new Point(0, 0)) );
					else
						mags.add( length(tmp.getP(), tmp2.getP()) );	
					tmp = tmp2; //rotate coordinate for next computation
				}
				
				
			}
		}
	}
}
