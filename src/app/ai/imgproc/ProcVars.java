package app.ai.imgproc;

import org.opencv.core.Scalar;

/**
 * tunable variables for image processing
 */
public class ProcVars {

    //debugging variables (draw in image where face and pupils are being detected
	public static boolean debug = true;
    public static final Scalar _COLOR = new Scalar(255, 255, 255); //color debugging shapes color
    public static final int _THICK = 2; //how thick the debugging shape should be
    //face detection variables
	static float mRelativeFS = 0.2f;
    static int mAbsoluteFS = 0;
	//eye region size constants - follows facial geometric rules
	static int kEyePercentTop = 25;
	static int kEyePercentSide = 13;
	static int kEyePercentHeight = 30;
	static int kEyePercentWidth = 35;
	//pupil detection algorithm params
	static double kFastEyeWidth = 40;
	static int kWeightBlurSize = 5;
	static boolean kEnableWeight =  false;
	static float kWeightDivisor = 1.0f;
	static double kGradientThresh = 50.0;
	//pupil post processing varaibles
	static double placeVal = 100.0; //used for rounding pupil coordinate percentages
    static double closedThresh = .98;

}
