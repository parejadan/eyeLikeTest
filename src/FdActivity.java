
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ArffLoader.ArffReader;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Date;

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
import app.ai.core.CoreVars;
import app.ai.core.LimQueue;

import java.util.Scanner;

public class FdActivity {

	public static void main(String[] args) throws Exception {
    	System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Detect obj = new Detect();

        obj.train();
        //obj.test();
        //obj.captureFromVidFile(15.33, 30, 60, 44, "s102/vid.wmv", "s102/labels.txt", "s102/training-data.arff");
        //obj.captureFromVidFile(15.33, 30, 60, 54, "s103/vid.wmv", "s103/labels.txt", "s103/training-data.arff");
        //obj.captureFromVidFile(15.33, 30, 71, 80, "s108/vid.wmv", "s108/labels.txt", "s108/training-data.arff");
        //obj.captureFromVidFile(15.33, 30, 61, 47, "s104/vid.wmv", "s104/labels.txt", "s104/training-data.arff");
        //obj.captureFromVidFile(15.33, 30, 24, 42, "s106/vid.wmv", "s106/labels.txt", "s106/training-data.arff");
        //obj.captureFromVidFile(15.33, 30, 60, 64, "s101/vid.wmv", "s101/labels.txt", "s101/training-data.arff");
        //obj.captureFromVidFile(15.33, 30, 60, 64, "s107/vid.wmv", "s107/labels.txt", "s107/training-data.arff");

    }

}

class Detect {
	static String trainFile = "dummy-data.arff";
    //APIs
    private static FindFace _faceD; //face detection object
    private static FindEyes _eyeD = new FindEyes(); //pupil detection object
    private static Stats _stool = new Stats(); //mathematics and stats object
    public static LimQueue<Double> mags;
    private static VideoCapture cam = new VideoCapture(0);
    FastVector fvWekaAttributes;
    static String[] labels = {"not-distracted", "distracted"}; // or non-distracted
    Scanner input = new Scanner(System.in);
	//PUPIL DETECTION OBJECTS
    Point last, cur; 
    ProcStruct ps;
	Mat frame  = new Mat();
	int frms = 0;

	Detect() {
        _faceD = new FindFace( new CascadeClassifier(getClass().getResource("/haarcascade_frontalface_alt.xml").getPath()) );
        
    	Attribute[] attributes;
    	attributes = new Attribute[CoreVars.queueSize+1];
        fvWekaAttributes = new FastVector(4); // Declare the feature vector
        
    	for (int i = 0; i < CoreVars.queueSize; i++) {
    		attributes[i] = new Attribute(String.format("magnitude%d", i) ); // create numeric attributes
    		fvWekaAttributes.addElement(attributes[i]);
    	}
        FastVector fvClassVal = new FastVector(2); // Declare the class attribute along with its values
        fvClassVal.addElement(labels[0]);
        fvClassVal.addElement(labels[1]);
        attributes[CoreVars.queueSize] = new Attribute("theClass", fvClassVal);
        fvWekaAttributes.addElement(attributes[CoreVars.queueSize]);
	}

	public double getFPS(String vidFile, double vidLen) {
		cam = new VideoCapture(vidFile);
		System.out.println("obtaining video fps");
		do {
			frms++;
			cam.read(frame);
		} while ( !frame.empty() );
		System.out.printf("recorded frames: %d\t|reported time: %f s\t|calculated FPS: %f\n", frms, vidLen, frms / vidLen);
		return frms / vidLen;
	}
	
    boolean readNextFrame() {
		cam.read(frame); //read in next frame
		
		if ( frame.empty() ) {
			mags.add(0.0);
			return false; //no more video feed
		}
		
    	ps = _faceD.getFace(frame);
        if (ps.getRect() == null) { //face cannot be located
            ps.setpupil( new org.opencv.core.Point(-1, -1) );
        } else { //locate right pupil within face region
            ps.setpupil(_eyeD.getPupil(ps.getImg(), ps.getRect(), true));
        }
    	cur = ps.getpupil();
    	Highgui.imwrite( "detection.png", ps.getImg() );
    	
        //treat closed eye signals as no eye movement
        if (cur.x  ==  -1 && cur.y == -1) {
        	mags.add(0.0);
            last = cur; //rotate coordinate for next computation
            return true; //there was video feed, but no pupil to extract
        } else if (last.x == -1) {
        	mags.add( _stool.length(0.0, 0.0, cur.x, cur.y) );
        } else if (cur.x == -1) {
        	mags.add( _stool.length(last.x, last.y, 0.0, 0.0) );
        } else {
        	mags.add( _stool.length(last.x, last.y, cur.x, cur.y) );
        }
        last = cur; //rotate coordinate for next computation
    	frame.release();
    	return true;
    }
      
    boolean fillUpQueue() {
    	mags = new LimQueue<Double>(CoreVars.queueSize); //reset mags
        
        //FILL UP THE MAGNITUDE QUEUE
		cam.read(frame); //read in an initial frame
    	ps = _faceD.getFace(frame);
        if (ps.getRect() == null) { //face cannot be located
            ps.setpupil( new org.opencv.core.Point(-1, -1) );
        } else { //locate right pupil within face region
            ps.setpupil(_eyeD.getPupil(ps.getImg(), ps.getRect(), true));
        }
        last = ps.getpupil();
        
    	//while ( mags.hasRoom() && readNextFrame() ); //false when queue is finally full or no more video feed
        while ( mags.hasRoom() ) {
        	readNextFrame();
        }
        
    	return !mags.hasRoom(); //true when queue is full, false otherwise
    }

    /**
     * @param vidFPS -  video file frames per second
     * @param stampT - time stamp relative to video that a classification occurs (in seconds)
     * @param vidLenMins - video length in minutes
     * @param startT - time in seconds where code should "fast forward" to in video
     * @param vidFile - video file name with file extension
     * @param lblsFile - manual classification data file with extension
     * @throws IOException
     */
    void captureFromVidFile(double vidFPS, int stampT, int vidLenMins, int startT, String vidFile, String lblsFile, String outFile) throws IOException {
    	trainFile = outFile;
    	cam = new VideoCapture(vidFile);
    	System.out.printf("Fast forwarding %ds into the video\n", startT);
    	for(int i = 0; i < (int) (startT*vidFPS); i++) cam.read(frame); //fast forward to where video should begin recording
    	
    	System.out.println("DATA COLLECTION BEGINNING");
    	CoreVars.queueSize = (int) (stampT*vidFPS);
    	CoreVars.rows = vidLenMins*60/stampT;
    	
    	System.out.printf("Queue Size is: %d\n", CoreVars.queueSize);
        
        Instances isTrainingSet = new Instances("Rel", fvWekaAttributes, CoreVars.rows); // Create an empty training set
        // Set class index
        isTrainingSet.setClassIndex(CoreVars.queueSize);

        Instance newReading;
    	input = new Scanner( new File(lblsFile) );
    	for (int i = 0; i < CoreVars.rows; i++ ) {
    		
    		if ( fillUpQueue() ) {
                newReading = new Instance(CoreVars.queueSize+1);
                for (int k = 0; k < CoreVars.queueSize; k++) newReading.setValue((Attribute) fvWekaAttributes.elementAt(k), mags.getVal(k));
                
                newReading.setValue((Attribute)fvWekaAttributes.elementAt(CoreVars.queueSize), labels[input.nextInt()] );
                isTrainingSet.add(newReading); // add the instance
    		}
			System.out.printf("Queue Size %d\n", mags.size());	
    	}
        saveArff(isTrainingSet, trainFile);
    	
    }

    void train() throws IOException  {
    	System.out.println("TRAINNING IS BEGINNING");
    	int i, j;
        // queueSize := training length; allocate equal room for attentive and inattentive
        Instances isTrainingSet = new Instances("Rel", fvWekaAttributes, CoreVars.rows*2); // Create an empty training set
        // Set class index
        isTrainingSet.setClassIndex(CoreVars.queueSize);

        Instance newReading;
        for (i = 0; i < labels.length; i++) {
            System.out.printf("Now Collecting %s dataset. Press enter to begin: ", labels[i]);
            input.nextLine();

            fillUpQueue();
            
            for (j = 0; j < CoreVars.rows; j++) {
                newReading = new Instance(CoreVars.queueSize+1);
                for (int k = 0; k < CoreVars.queueSize; k++) newReading.setValue((Attribute)fvWekaAttributes.elementAt(k), mags.getVal(k));
                
                newReading.setValue((Attribute)fvWekaAttributes.elementAt(CoreVars.queueSize), labels[i]);
                isTrainingSet.add(newReading); // add the instance
                readNextFrame();
            }
        }

        saveArff(isTrainingSet, trainFile);
    }
    
    void test() throws Exception {
    	System.out.println("TESTING IS BEGINNING");
    	Instances isTrainingSet = loadArff(trainFile);
    	isTrainingSet.setClassIndex(isTrainingSet.numAttributes() - 1);

    	// After training data set is complete, build model
        Classifier cModel = (Classifier)new RandomForest();
        cModel.buildClassifier(isTrainingSet);
        Evaluation eTest = new Evaluation(isTrainingSet);
                
        // Time variables
        long cTime, sTime, secondsWait = 5l;
        boolean wasDistracted = false;
        double isDistracted;
        Instances dataUnlabeled = new Instances("TestInstances", fvWekaAttributes, 0);
        Instance newFrame = new Instance(CoreVars.queueSize+1);
        
        // Assign whatever as the class - this isn't looked at while being evaluated
        newFrame.setValue((Attribute)fvWekaAttributes.elementAt(CoreVars.queueSize), labels[0]);
        dataUnlabeled.add(newFrame);
        dataUnlabeled.setClassIndex(dataUnlabeled.numAttributes() - 1);

        System.out.print("Press enter to begin: ");
        input.nextLine();

        //fillUpQueue();
        sTime = new  Date().getTime();
        //while(true) {
        for(int i = 0; i < CoreVars.rows; i++) {
            fillUpQueue();
        	for (int k = 0; k < CoreVars.queueSize; k++) newFrame.setValue((Attribute)fvWekaAttributes.elementAt(k), mags.getVal(k));

            // Predict
            isDistracted = eTest.evaluateModelOnceAndRecordPrediction(cModel, dataUnlabeled.lastInstance());
            cTime = new Date().getTime();

            // Also throw in a check for whether driver has been alerted in last ~5s
            if (isDistracted == 1 && wasDistracted  && (cTime - sTime) / 1000l >= secondsWait) {
            	System.out.print( (cTime - sTime) );
            	System.out.println("\t" + labels[1]); // Turn this into an alert on the phone
            	sTime = cTime; // update most recent alert
            }
            // Set up for next frame
            if (isDistracted == 1) wasDistracted = true;
            else wasDistracted = false;

            readNextFrame();
        }
    }

    /* Saves the Instances object as an .arff file.
     * Pre-conditions:
     *  location specifies an absolute path.
     * Post-conditions:
     *  saves dataSet at location.
     */
    void saveArff(Instances dataSet, String saveFile) throws IOException {
		 ArffSaver saver = new ArffSaver();
		 saver.setInstances(dataSet);
		 saver.setFile(new File(saveFile));
		 saver.writeBatch();
    }
    
    /* Loads an Instances object from an .arff file.
     * Pre-conditions:
     *  location specifies an absolute path.
     * Post-conditions:
     *  loads Instances from location.
     */
    Instances loadArff(String location) throws IOException {
    	BufferedReader reader = new BufferedReader(new FileReader(location));
    	ArffReader arff = new ArffReader(reader);
    	return arff.getData();
    }

}