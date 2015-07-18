


import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;

import app.ai.imgproc.FindEyes;
import app.ai.imgproc.FindFace;
import app.ai.imgproc.ProcStruct;
import app.ai.stats.Stats;
import app.ai.core.LimQueue;
import app.ai.core.CoreVars;

public class FdActivity {

	public static void main(String[] args) {
    	System.loadLibrary(Core.NATIVE_LIBRARY_NAME);    	
        new Detect().collectPoints();
    }
    
}

class Detect {
	
    //data collection structure
    private static LimQueue<Point> trainingPnts = new LimQueue<Point>(CoreVars.queueSize+1);
    //APIs
    private static FindFace _faceD; //face detection object
    private static FindEyes _eyeD; //pupil detection object
    private static Stats _stool = new Stats(); //mathematics and stats object
	
    void collectPoints() {
    	VideoCapture cam = new VideoCapture(0);
    	CascadeClassifier ccf = new CascadeClassifier(getClass().getResource("/haarcascade_frontalface_alt.xml").getPath());
        _faceD = new FindFace( ccf );
        _eyeD = new FindEyes();
        Mat frame = new Mat();

        while ( trainingPnts.hasRoom() ) { //read in frames until training fully collected
			cam.read(frame);
			
	        ProcStruct ps = _faceD.getFace(frame);

	        if (ps.getRect() == null) { //face cannot be located
	            ps.setpupil( new org.opencv.core.Point(-1, -1) );
	            trainingPnts.add(ps.getpupil());

	        } else { //locate right pupil within face region
	            ps.setpupil(_eyeD.getPupil(ps.getImg(), ps.getRect(), false));
	            trainingPnts.add( ps.getpupil() );
	        }

	    	Highgui.imwrite( "detection.png", ps.getImg() );
        }

    }
    
    void train()  {
        Point tmp, tmp2;
        int i, size = trainingPnts.size();
        tmp = trainingPnts.getVal(0);
        ArrayList<Double> mags = new ArrayList<Double>();

        for (i = 1; i < size; i++) {
            tmp2 = trainingPnts.getVal(i);

            //treat closed eye signals as no eye movement
            if (tmp.x == -1) mags.add( _stool.length(0.0, 0.0, tmp2.x, tmp2.y) );
            else if (tmp2.x == -1) mags.add( _stool.length(tmp.x, tmp.y, 0.0, 0.0) );
            else if (tmp.x  ==  -1 && tmp2.y == -1) {
                tmp = tmp2; //rotate coordinate for next computation
                continue;
            }
            else mags.add( _stool.length(tmp.x, tmp.y, tmp2.x, tmp2.y) );

            tmp = tmp2; //rotate coordinate for next computation
        }
        
        
        System.out.println("Processing complete..");
        System.out.printf("recorded frames: %d\t| obtained magnitudes: %d", trainingPnts.size(), mags.size());
        System.out.printf("mean from training data: %f", _stool.getMean(mags));

    }
}