
import org.opencv.core.Point;

import java.util.ArrayList;

class Recs {
	private long tLog;
	private Point xy;
	
	public Recs(Point pt, long time) {
		xy = pt;
		tLog = time;
	}
	
	public long getT() { return tLog; }
	public Point getP() { return xy; }
}

public class Records {
	ArrayList<Recs> data;
	private long expTime;
	int size;
	
	public Records(long expT) {
		data = new ArrayList<Recs>();
		expTime = expT;
		size = 0;
	}
	
	public void add (Point pt, long curTime) {
		data.add( new Recs(pt, curTime) ); //add new record
		//remove out dated frames that exceed expiration time
		Recs tmp;
		while (data.size() > 0) {
			tmp = data.get(0); //old frames are located at the beginning of the list
			if ( curTime - tmp.getT() >= expTime ) data.remove(0);
			else break;
		}
	}
	
	public ArrayList<Recs> getDat() { return data; } //java returns a reference
	
	public int size() { return data.size(); }
}
