
import org.opencv.core.Point;

import java.util.ArrayList;

public class Records {
	ArrayList<Point> data;
	private int expLength;
	
	public Records(int expL) {
		data = new ArrayList<Point>();
		expLength = expL;
	}
	
	public void add (Point pt) {
		if (data.size()+1 > expLength)
			data.remove(0);
		data.add(pt);	
	}

}
