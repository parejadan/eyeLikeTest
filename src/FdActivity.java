
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.highgui.Highgui;
import org.opencv.objdetect.CascadeClassifier;
//import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

class DetectFaceDemo {
    private static Mat                    mGray = new Mat();
    private static CascadeClassifier      mJavaDetector; //java classifier
    private static String 				  img = "img1.png";
    
    
    private static FindEyes fd = new FindEyes();
    private static org.opencv.core.Point[] pupils;

    public void run() {
    	mJavaDetector = new CascadeClassifier(getClass().getResource("/haarcascade_frontalface_alt.xml").getPath()); //works with gray scale images
    	//load image then gray scale it for detection
    	ArrayList<Mat> mv = new ArrayList<Mat>(3);
    	Core.split( Highgui.imread(getClass().getResource(img).getPath()) , mv);
    	mGray = mv.get(2); //use the lightest gray scale
    	
    	MatOfRect faceDetections = new MatOfRect();
    	mJavaDetector.detectMultiScale(mGray, faceDetections);
    	
    	//for (Rect rect : faceDetections.toArray() ) //draw detected faces on image
    		//Core.rectangle(mGray, rect.tl(), rect.br(), Vars.FACE_RECT_COLOR); //.tl()-> x & y points, .br() -> img (w, h)+(x,y)

    	if (faceDetections.toArray().length > 0) { //rectangles for each face detected
    		for (Rect rect : faceDetections.toArray() ) {
        		pupils = fd.detect(mGray, rect); //detect pupils
        		Core.circle(mGray,  pupils[0],  2,  Vars.EYE_CIRCLE_COLR); //draw left
        		Core.circle(mGray,  pupils[1],  2,  Vars.EYE_CIRCLE_COLR); //draw right
    		}
    	}

    	Highgui.imwrite("detection.png", mGray);
    }
}

public class FdActivity {
    public static void main(String[] args) {
    	System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
    	new DetectFaceDemo().run();
    }
}