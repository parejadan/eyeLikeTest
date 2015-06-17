import org.opencv.core.Core;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;


public class FindEyes {
	private int eRows, eCols;
	
	private Helpers hp = new Helpers();
	
	public FindEyes() { }
		
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
	
/*	
	void testPossibleCenterFormula(int x, int y, Mat weight, double gx, double gy, Mat out) {
		int rows = weight.rows()-1, cols = weight.cols()-1;
		double dx, dy, mag, dot;
		double[] bufO;
		for (int j = 0; j < rows; j++) {
			for (int k = 0; k < cols; k++) {
				//avoid a specific location
				if (j == y && k == x) continue;
				
				bufO = out.get(j, k);
				dx = x - k;
				dy = y - j;
				mag = Math.sqrt(dx*dx + dy*dy);
				dx = dx/mag;
				dy = dy/mag;
				
				dot = Math.max(0.0, dx*gx + dy*gy);

				if (Vars.kEnableWeight)
					bufO[0] += dot*dot *(weight.get(j, k)[0]/Vars.kWeightDivisor);
				else
					bufO[0] += dot*dot;
				//System.out.printf("%f\t|%f\t|%b\t|%f\n", dx*gx + dy*gy, dot, dx*gx + dy*gy < 0,(double) bufO[0]);
				//if (bufO[0] < 0)
					//System.out.println("negs");
				//else
					//System.out.printf("%f\n", bufO[0]);
				out.put(j, k, bufO);
				//System.out.printf("%f\n", out.get(j, k)[0]);
			}
		}
	}*/

	void testPossibleCenterFormula(int x, int y, Mat weight, double gx, double gy, Mat out) {
		int rows = weight.rows()-1, cols = weight.cols()-1;
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
	
	Mat computeGradient(Mat mat) {
		int rows = mat.rows()-1, cols = mat.cols()-1;
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
		//Core.rectangle(face, eye.tl(), eye.br(), Vars.FACE_RECT_COLOR); //draw eye region - for debugging
    	//find the gradient
		Mat gradX = computeGradient(eyeROI);
		Mat gradY = computeGradient(eyeROI.t()).t();
		//compute all pixel magnitudes
		Mat mags = hp.matrixMagnitude(gradX, gradY);
		gradientThresh = hp.computeDynamicThreshold(mags, Vars.kGradientThresh); //compute all pixel threshold
		//normalize
		int rows  = eyeROI.rows()-1, cols = eyeROI.cols()-1;
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
		return unscalePoint(mmr.maxLoc, eye);
	}

	Point[] detect(Mat frame_gray, Rect face) {    	
		Mat faceROI = frame_gray.submat(face);

		if (Vars.kSmoothFaceImage)
			Imgproc.GaussianBlur(faceROI, faceROI, (new Size(0, 0)), Vars.kSmoothFaceFactor * face.width);
		//determine a face to eye ratio and positioning based on percentages
		int eyeRegW = (int) (face.width * (Vars.kEyePercentWidth/100.0));
		int eyeRegH = (int) (face.height * (Vars.kEyePercentHeight/100.0));
		int eyeDexY = (int) (face.height * (Vars.kEyePercentTop/100.0));
		int eyeDexX = (int) (face.width * (Vars.kEyePercentSide/100.0));		
		Rect lEyeReg = new Rect(eyeDexX, eyeDexY, eyeRegW, eyeRegH);
		Rect rEyeReg = new Rect(face.width - eyeRegW - eyeDexX, eyeDexY, eyeRegW,eyeRegH);
		//calculate rectangle origins for face and eyes
		Point cFace = new Point(face.width/2.0 + face.x, face.height/2.0 + face.y);
		// 4 and 1 accommodate minor int arithmetic value lost in calculations
		Point cEyeL = new Point(cFace.x - lEyeReg.width - 4, cFace.y - lEyeReg.y - 1);
		Point cEyeR = new Point(cFace.x + lEyeReg.width + 4, cFace.y - rEyeReg.y - 1);
		//detect pupils for left eye region (the camera's left, not the model's)
		Point[] pupils = { new Point(0,0), new Point(0,0) };
		
		pupils[0] = findEyeCenter(faceROI,lEyeReg);
		pupils[0].x += cEyeL.x;
		pupils[0].y += cEyeL.y;
		//detect pupils for right  eye region (the camera's right, not the model's)
		pupils[1] = findEyeCenter(faceROI,rEyeReg);
		pupils[1].x = cEyeR.x - pupils[1].x;
		pupils[1].y += cEyeR.y;
		
		return pupils;
	}
}
