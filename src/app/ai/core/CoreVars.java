package app.ai.core;

public class CoreVars {

	//training and monitoring variables
    public static int queueSize = 100; //460; //how many magnitudes are required for training and monitoring
    public static int rows = 700;
    static int alphaCount = queueSize; //how many alphas we should simulate
    static double subsize = 2/3.0; //percent of queue size data is subsampled for computing alphas
    static double dev, key; //standard derivation and mean of training data
    boolean magFLag; //true when eyes movement deviates from baseline training data
    public static int maxWidth = 400;
    public static int maxHeight = 400;

}
