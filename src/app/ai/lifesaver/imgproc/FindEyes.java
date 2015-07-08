/**
 * This file contains the code used for eye detection. Face must already 
 *  be located and passed to detect() parameter
 */

package app.ai.lifesaver.imgproc;

import org.opencv.core.Core;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Mat;
//import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;

public class FindEyes {
	public static int eyeRegW, eyeRegH;
	public static double[] cEye;
	private static int rows, cols;
	
	private Helpers hp = new Helpers();
	
	public FindEyes() {
		cEye = new double[2];
	}
		
	public Point unscalePoint(Point p, Rect origSize) {
		float ratio = ( (float) Vars.kFastEyeWidth/origSize.width );
		int x = (int) (p.x / ratio);
		int y = (int) (p.y / ratio);
		return (new Point(x, y) );
	}
	
	public  Mat scaleToFastSize(Mat src) {
		Mat resized = src.clone();
		if (resized.cols() > 0)
		Imgproc.resize(src, resized, new Size(Vars.kFastEyeWidth, Vars.kFastEyeWidth/src.cols() * src.rows()));
			
		return resized;
	}

	void testPossibleCenterFormula(int x, int y, Mat weight, double gx, double gy, Mat out) {
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
				if (Vars.kEnableWeight) bufO[0] += dot*dot *(weight.get(j, k)[0]/Vars.kWeightDivisor);
				else bufO[0] += dot*dot;
				
				out.put(j, k, bufO);	
			}
		}
	}
	
	Mat computeGradient(Mat mat, int rows, int cols) {
		Mat out = Mat.zeros(rows, cols, CvType.CV_64F);
		
		for (int y = 0; y < rows; y++) {
			out.put( y, 0, mat.get(y, 1)[0] - mat.get(y, 0)[0] );
			
			for (int x = 1; x < cols - 1; x++)
				out.put( y, x, (mat.get(y, x+1)[0] - mat.get(y, x-1)[0]) / 2.0 );
			
			out.put(y, cols-1, mat.get(y, cols-1)[0] - mat.get(y, cols-2)[0]);
		}
		return out;
	}

	public Point findEyeCenter(Mat face, Rect eye) {
		double gx, gy, gradientThresh;
		Mat eyeROI = scaleToFastSize( face.submat(eye) );
    	//Highgui.imwrite("eyeROI.png", eyeROI);
		//Highgui.imwrite("transpose.png", eyeROI.t());
		rows = eyeROI.rows()-1; cols = eyeROI.cols()-1;
		
		Core.rectangle(face, eye.tl(), eye.br(), Vars.FACE_RECT_COLOR); //draw eye region - for debugging
    	//find the gradient
		Mat gradX = computeGradient(eyeROI, rows, cols);
		Mat gradY = computeGradient(eyeROI.t(), cols, rows).t();
		//compute all pixel magnitudes
		Mat mags = hp.matrixMagnitude(gradX, gradY, rows, cols);
    	//Highgui.imwrite("mags.png", mags);
		gradientThresh = hp.computeDynamicThreshold(mags, Vars.kGradientThresh); //compute all pixel threshold
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
		Imgproc.GaussianBlur(eyeROI, weight, (new Size(Vars.kWeightBlurSize, Vars.kWeightBlurSize)), 0, 0);
    	//Highgui.imwrite("weight.png", weight);

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
    	//Highgui.imwrite("outSum.png", outSum);

		//find the maximum point
		Core.MinMaxLocResult mmr = Core.minMaxLoc(outSum);
		return unscalePoint(mmr.maxLoc, eye);
		//Point e = unscalePoint(mmr.maxLoc, eye);
		
		//outSum.release(); weight.release(); gradX.release(); gradY.release(); mags.release();
		//outSum = weight = gradX = gradY = mags = null;
		//return e;
	}
		
	public Point findPupil(Mat frame_gray, Rect face, boolean leftEye) {
		Mat faceROI = frame_gray.submat(face);
		Point pupil = null;
		Core.rectangle(frame_gray, face.tl(), face.br(), Vars.FACE_RECT_COLOR); //draw eye region - for debugging
		//faceOLD = frame.clone();
		/*if (Vars.kSmoothFaceImage)
			Imgproc.GaussianBlur(faceROI, faceROI, (new Size(0, 0)), Vars.kSmoothFaceFactor * face.width);*/

		//determine a face to eye ratio and positioning based on percentages
		eyeRegW = (int) (face.width * (Vars.kEyePercentWidth/100.0));
		eyeRegH = (int) (face.height * (Vars.kEyePercentHeight/100.0));
		int eyeDexY = (int) (face.height * (Vars.kEyePercentTop/100.0));
		int eyeDexX = (int) (face.width * (Vars.kEyePercentSide/100.0));
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
		//Core.circle(frame_gray, new Point(cEye[0], cEye[1]), 2, Vars.EYE_CIRCLE_COLR);
		//Core.circle(frame_gray, new Point(eyeDexX+face.x, eyeDexY+face.y), 2, Vars.EYE_CIRCLE_COLR);
		//Core.circle(frame_gray, new Point(face.x, face.y), 2, Vars.EYE_CIRCLE_COLR);
		//System.out.println(1.0*(cEye[1]-face.y)/face.height);

		faceROI.release(); faceROI = null;
		return pupil;
	}
}