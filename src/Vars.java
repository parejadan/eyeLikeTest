import org.opencv.core.Scalar;

/**
 * Constant variables to be used by the rest of the code.
 *  Pupil recognition can be fine-tuned by manipulating these values
 */
public class Vars {
    static final Scalar FACE_RECT_COLOR = new Scalar(255, 255, 355);
    static final Scalar EYE_CIRCLE_COLR = new Scalar(255,  255, 255);
	//size constants
	static int kEyePercentTop = 25;
	static int kEyePercentSide = 13;
	static int kEyePercentHeight = 30;
	static int kEyePercentWidth = 35;
	//pre-processing
	static boolean kSmoothFaceImage = false;
	static float kSmoothFaceFactor = 0.005f;
	//algorithm params
	static double kFastEyeWidth = 50;
	static int kWeightBlurSize = 5;
	static boolean kEnableWeight =  false;
	static float kWeightDivisor = 1.0f;
	static double kGradientThresh = 50.0;
}
