package app.ai.imgproc;

import org.opencv.core.Rect;
import org.opencv.core.Mat;
import org.opencv.core.Point;

public class ProcStruct {
	private static Rect face;
	private static Mat img;
	private static Point pupil;
    
	public ProcStruct() { };
	public ProcStruct(Mat i, Rect r) {
		face = r;
		img = i;
		pupil = null;
	}
	public void setRect(Rect r) {face = r; }
	public void setImg(Mat i) {
		if (img != null) img.release(); //address any possible memory leakage
		img = i;
	}
	
	public void setpupil(Point p) {
		
		if (p.x == -1) { //flag if face was not face detected
			pupil = p;
		} else if ( FindEyes.cEye[1]/p.y >= ProcVars.closedThresh ) { //flag incorrect eye detection ("if" true when eyes are closed)
    		pupil = new Point(-1, -1);
    	} else { //store detected eyes
    		pupil = p;
    		//mskpupilToEyereg();
    	}
		
	}
	
	public void mskpupilToEyereg() { //represent pupil coordinate as a ratio relative to eye face
		if (pupil.x > FindEyes.eyeRegW ) { //make sure coordinates are relative to eye region
    		pupil.x -= FindEyes.cEye[0];
    		pupil.y -= FindEyes.cEye[1];
		}
		
		pupil.x = (pupil.x / FindEyes.eyeRegW) * ProcVars.placeVal;
		pupil.y = (pupil.y / FindEyes.eyeRegW) * ProcVars.placeVal;

	}
	
	public Rect getRect() { return face; }
	public Mat getImg() { return img; }
	public Point getpupil() { return pupil; }
}
