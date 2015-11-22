
package app.ai.imgproc;

import org.opencv.core.Core;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;

public class FindEyes {

	//private final String eyeTAG = getClass().getSimpleName();;
	
	public static int eyeRegW, eyeRegH; //width and height of eye region
	public static double[] cEye; //coordinates of eyes center (relative to overall image)
	
	private static int rows, cols;
	private static Helpers _hp = new Helpers();
	
	public FindEyes() { cEye = new double[2]; }
		
	public Point unscalePoint(Point p, Rect origSize) {
		float ratio = ( (float) ProcVars.kFastEyeWidth/origSize.width );
		int x = (int) (p.x / ratio);
		int y = (int) (p.y / ratio);
		return (new Point(x, y) );
	}
	
	public Mat scaleToFastSize(Mat src) {
		Mat resized = src.clone();
		if (resized.cols() > 0)
		Imgproc.resize(src, resized, new Size(ProcVars.kFastEyeWidth, ProcVars.kFastEyeWidth/src.cols() * src.rows()));
			
		return resized;
	}

	public Mat computeGradient(Mat mat, int rows, int cols) {
		Mat out = Mat.zeros(rows, cols, CvType.CV_64F);
		
		for (int y = 0; y < rows; y++) {
			out.put( y, 0, mat.get(y, 1)[0] - mat.get(y, 0)[0] );
			
			for (int x = 1; x < cols - 1; x++)
				out.put( y, x, (mat.get(y, x+1)[0] - mat.get(y, x-1)[0]) / 2.0 );
			
			out.put(y, cols-1, mat.get(y, cols-1)[0] - mat.get(y, cols-2)[0]);
		}
		return out;
	}
	
	public void testPossibleCenterFormula(int x, int y, Mat weight, double gx, double gy, Mat out) {
		// Precompute values
		int xy_sum = x*x + y*y, y_double = 2*y, x_double = 2*x;
		double gxy = gx*x + gy*y, dot, top;
		double[] bufO;
		for (int j = 0; j < rows; j++) {
			for (int k = 0; k < cols; k++) {
				//avoid a specific location
				if (j == y && k == x) continue;

				// see if the result will be negative
				top = gxy - gx*k - gy*j;
				if (top <= 0) dot = 0; 
				else dot = top / Math.sqrt(xy_sum + k*(k - x_double) + j*(j - y_double));

				bufO = out.get(j, k);
				if (ProcVars.kEnableWeight) bufO[0] += dot*dot *(weight.get(j, k)[0]/ProcVars.kWeightDivisor);
				else bufO[0] += dot*dot;
				
				out.put(j, k, bufO);	
			}
		}
	}
	
	/**
	 * Pupil detection algorithm; passed image should be scaled down to just the face region.
	 * @param face - image containing only a face
	 * @param eye - rectangle that dictates where eye region is within passed face image
	 * @return - pupil x&y coordinate relative to eye region (NOT face region)
	 */
	public Point findEyeCenter(Mat face, Rect eye) {
		double gx, gy, gradientThresh;
		Mat eyeROI = scaleToFastSize( face.submat(eye) );
		rows = eyeROI.rows()-1; cols = eyeROI.cols()-1;
		
		if (ProcVars.debug) Core.rectangle(face, eye.tl(), eye.br(), ProcVars._COLOR); //draw eye region - for debugging
    	
		//find the gradient
		Mat gradX = computeGradient(eyeROI, rows, cols);
		Mat gradY = computeGradient(eyeROI.t(), cols, rows).t();
		//compute all pixel magnitudes
		Mat mags = _hp.matrixMagnitude(gradX, gradY, rows, cols);
		gradientThresh = _hp.computeDynamicThreshold(mags, ProcVars.kGradientThresh); //compute all pixel threshold
		//normalize
		for (int y = 0; y < rows; y++)
			for (int x = 0; x < cols; x++) {
				if (mags.get(y, x)[0] > gradientThresh) {
					gradX.put(y, x, gradX.get(y, x)[0]/mags.get(y, x)[0]);
					gradY.put(y, x, gradY.get(y, x)[0]/mags.get(y, x)[0]);
				} else {
					gradX.put(y, x, 0.0);
					gradY.put(y, x, 0.0);
				}
			}
		//create a blurred and inverted image for weighting
		Mat weight = new Mat();
		Imgproc.GaussianBlur(eyeROI, weight, (new Size(ProcVars.kWeightBlurSize, ProcVars.kWeightBlurSize)), 0, 0);

		for (int y = 0; y < rows+1; y++)
			for (int x = 0; x < cols+1; x++) {
				double[] tmp =  weight.get(y, x);
				tmp[0] = 255 - tmp[0];
				weight.put(y, x, tmp);
			}
		//run algorithm
		Mat outSum = Mat.zeros(rows, cols, CvType.CV_64F);
		for (int y = 0; y < rows; y++)
			for (int x = 0; x <  cols; x++) {
				gx = gradX.get(y, x)[0]; gy = gradY.get(y, x)[0];
				if (gx == 0.0 && gy == 0.0) {
					continue;
				}
				testPossibleCenterFormula(x, y, weight, gx, gy, outSum);
			}

		//find the maximum point
		Core.MinMaxLocResult mmr = Core.minMaxLoc(outSum);
		eyeROI.release(); gradX.release(); gradY.release(); mags.release(); weight.release(); outSum.release();
		
		return unscalePoint(mmr.maxLoc, eye);
	}
	
	/**
	 * Locates pupils for a given face within an image; assumes the image contains a face
	 * 
	 * @param frame_gray - image processed to the highest possible grey scale 
	 * @param face - face within image that module must locate pupil in
	 * @param leftEye - true if user wants the left pupil, false otherwise
	 * @return x & y coordinates relative to image
	 */
	public Point getPupil(Mat frame_gray, Rect face, boolean leftEye) {
		Mat faceROI = frame_gray.submat(face);
		Point pupil = new Point(-1, -1);

		if (ProcVars.debug) Core.rectangle(frame_gray, face.tl(), face.br(), ProcVars._COLOR); //draw eye region - for debugging

		//determine a face to eye ratio and positioning based on percentages
		eyeRegW = (int) (face.width * (ProcVars.kEyePercentWidth/100.0));
		eyeRegH = (int) (face.height * (ProcVars.kEyePercentHeight/100.0));
		int eyeDexY = (int) (face.height * (ProcVars.kEyePercentTop/100.0));
		int eyeDexX = (int) (face.width * (ProcVars.kEyePercentSide/100.0));
		Rect lEyeReg = new Rect(eyeDexX, eyeDexY, eyeRegW, eyeRegH);
		double[] cFace = {face.width/2.0 + face.x, face.height/2.0 + face.y};

		if (leftEye) { //detect pupils for left eye region (the camera's left, not the model's)
			// 4 and 1 accommodate minor int arithmetic value lost in calculations
			cEye[0] = cFace[0] - lEyeReg.width - 4; cEye[1] = cFace[1] - lEyeReg.y - 1;
			pupil = findEyeCenter(faceROI,lEyeReg);
			pupil.x += cEye[0];
			pupil.y += cEye[1];

		} else {
			Rect rEyeReg = new Rect(face.width - eyeRegW - eyeDexX, eyeDexY, eyeRegW,eyeRegH);
			cEye[0] = cFace[0] + 4; cEye[1] = cFace[1] - rEyeReg.y - 1;
			pupil = findEyeCenter(faceROI,rEyeReg);
			pupil.x += cEye[0];
			pupil.y += cEye[1];
		}
		
		if (ProcVars.debug) Core.circle(frame_gray, pupil, ProcVars._THICK, ProcVars._COLOR);
		faceROI.release();
		
		return pupil;
	}

}