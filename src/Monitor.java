

import java.util.ArrayList;

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
import app.ai.lifesaver.stats.Stats;

public class Monitor {

    private static FindEyes fd = new FindEyes(); //pupil detection interface
    private static Stats st = new Stats(); //statistical tools interface
    private static CascadeClassifier mJavaDetector; //java classifier
    private static float mRelativeFS = 0.2f;
    private static int mAbsoluteFS = 0;
    private static double closedThresh;
    private static double placeVal;

    private static double std, key;
    private static int trnlen, checklen;
    private static int picksize, subsize; 
    boolean cloFlag;
    boolean magFlag;
    
    /**
     * @param mJDetector - cascade classifier for face detection
     * @param ct - percent pupil 
     * @param at - acceptable time eyes can remain closed or not present (milliseconds)
     */
    public Monitor(CascadeClassifier mJDetector, double ct, double pVal, int trs) {
    	mJavaDetector = mJDetector;
    	closedThresh = ct;
    	placeVal = pVal;
    	trnlen = trs;
    	checklen = trnlen;
    	subsize = picksize = 2*checklen/3;
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
		
    	Highgui.imwrite("detection.png", mGray);

        for (Mat tmp : mv) tmp.release();
        mv.clear();
        faces.release(); mGray = faces = null;
        
        return pupil;
	}

	double length(Point a, Point b) { return (int) Math.sqrt( Math.pow(a.x-b.x, 2.0) + Math.pow(a.y-b.y, 2.0) ); }
	
	/** Checks if the eye movement falls outside a standard deviation interval from target eye movement
	 * @param mag - magnitude to check on
	 * @return true if it falls outside it, false otherwise
	 */
	boolean checkMovement(double mag) { return (mag < key-std || mag > key+std) ? true : false; }

	void train(VideoCapture cam) {
		ArrayList<Point> tracking = new ArrayList<Point>();
		Mat frame = new Mat();
		Point tmp, tmp2;
		int i;
		
		//record frames
		for (i = 0; i < trnlen; i++) {
			cam.read(frame);
			tracking.add(getPupilxy(frame));
			frame.release();
		}
		
		//get baseline Key and std from training data
		ArrayList<Double> mags = new ArrayList<Double>(), moments;
		tmp = tracking.get(trnlen-1);
		for (i = trnlen-2; i >= 0; i--) { //compute magnitude with frames
			tmp2 = tracking.get(i);
			
			//treat closed eye signals as no eye movement
			if (tmp.x == -1) mags.add( length(new Point(0, 0), tmp2) );
			else if (tmp2.x == -1) mags.add( length(tmp, new Point(0, 0)) );
			else mags.add( length(tmp, tmp2) );	
			
			tmp = tmp2; //rotate coordinate for next computation
		}
		System.out.printf("mags: %d\t|trnlen: %d\t|checklen: %d\n", mags.size(), trnlen, checklen);
		moments = st.genRandMoments(mags, picksize, subsize);
		key = st.getMean( moments );
		std = st.getSTD(moments, key);

		//i = 0;
		//for (i = mags.size()-1; i >= 0; i--)
			//System.out.printf("%f%s", mags.get(i), (i == 0) ? "\n":",");
		
	}
	
	/**
	 * Monitors drivers pupil movement to see if they are acceptably active or not.
	 * cloFlag - true whenever a driver is considered to have eyes closed for an extended period
	 * magFlag - true whenever a drivers eye movement falls bellow an acceptable baseline (obtained from training)
	 * @param cam - pointer to video camera where frames can be extracted from
	 */
	void watch(VideoCapture cam) {
		Records tracking = new Records(checklen);
		ArrayList<Double> mags, moments;
		Mat frame = new Mat();
		Point tmp, tmp2;
		double mean;

		for (int j = 0; true; j++) {
			cam.read(frame);
			tracking.add(getPupilxy(frame));
			frame.release();
			
			if (j >= checklen) { //if we have the trainning amount of frames
				mags = new ArrayList<Double>();
				tmp = tracking.data.get(checklen-1); //get the very last frame on list (considered the most recent one recorded)
				for (int i = checklen-2; i >= 0; i--) {
					tmp2 = tracking.data.get(i);
					
					//treat closed eye signals as no eye movement
					if (tmp.x == -1) mags.add( length(new Point(0, 0), tmp2) );
					else if (tmp2.x == -1) mags.add( length(tmp, new Point(0, 0)) );
					else mags.add( length(tmp, tmp2) );	
					
					tmp = tmp2; //rotate coordinate for next computation
				}
				
				moments = st.genRandMoments(mags, picksize, subsize);
				mean = st.getMean( moments );
				//sig = st.getSTD(mags, mean);
				magFlag = checkMovement(mean);
				
				System.out.printf("train-m: %f\t|check-m: %f\t\n", key, mean);
				System.out.printf("train-s: %f\t\n", std);
				System.out.printf( "%s\n\n", (magFlag) ? "distracted" : "paying attention");
			}

		}
	}
}
