
import org.opencv.core.Core;
//import org.opencv.core.Mat;
//import org.opencv.core.MatOfRect;
//import org.opencv.core.Rect;
//import org.opencv.highgui.Highgui;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.highgui.VideoCapture;
//import org.opencv.imgproc.Imgproc;

//import app.ai.lifesaver.imgproc.*;

//import java.util.ArrayList;

class DetectFaceDemo {
    //private static Mat                    mGray = new Mat();
    private static CascadeClassifier      mJavaDetector; //java classifier    
    
    //private static FindEyes fd = new FindEyes();
    //private static org.opencv.core.Point pupil;

    public void monitor() {
    	
    	VideoCapture cam = new VideoCapture(0);
    	mJavaDetector = new CascadeClassifier(getClass().getResource("/haarcascade_frontalface_alt.xml").getPath()); //works with gray scale images
    	
    	Monitor mon = new Monitor(mJavaDetector, .98, 100.0, 50, 1, 100); //250 magnitudes for training (500 frames)
    	System.out.println("traning beginning");
    	mon.train(cam);
    	System.out.println("traning complete");
    	//System.out.println("beginning monitoring");
    	//mon.watch(cam);
    	/***
    	//Mat frame = Highgui.imread( getClass().getResource("ascent-p26.bmp").getPath() );
    	//Rect tmp = new Rect(0, 0, frame.width(), frame.height() );
    	//fd.findEyeCenter(frame, tmp);
    	Mat frame = new Mat();
    	ArrayList<Mat> mv = new ArrayList<Mat>(3);
    	MatOfRect faceDetections = new MatOfRect();
    	while (true) {
    		cam.read(frame);
        	Core.split(frame , mv);
        	mGray = mv.get(2); //use the lightest gray scale
        	
        	mJavaDetector.detectMultiScale(mGray, faceDetections);
        	
        	for (Rect rect : faceDetections.toArray() ) //draw detected faces on image
        		Core.rectangle(mGray, rect.tl(), rect.br(), Vars.FACE_RECT_COLOR); //.tl()-> x & y points, .br() -> img (w, h)+(x,y)

        	if (faceDetections.toArray().length > 0) { //rectangles for each face detected
        		for (Rect rect : faceDetections.toArray() ) {
            		pupil = fd.findPupil(mGray, rect, true); //detect pupil right pupil by default
                	//if ( mGray.get((int) pupil.y, (int) pupil.x)[0] > 135 )
                		//System.out.println("eyes closed");

            		Core.circle(mGray,  pupil,  5,  Vars.EYE_CIRCLE_COLR);
        		}
        	}

        	Highgui.imwrite("detection.png", mGray);
    		mGray.release();
    		for (Mat tmp : mv) {
    			tmp.release();
    			tmp = null;
    		}
    		frame.release();
    	}*/
    }
}

public class FdActivity {
    public static void main(String[] args) {
    	System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    	new DetectFaceDemo().monitor();
    }
}