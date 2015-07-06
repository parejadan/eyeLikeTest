import java.awt.BorderLayout;
import java.awt.Color;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class FeedBack extends JFrame {
	private JPanel green = new JPanel();
	private JPanel red = new  JPanel();
	int state; //true when red, false when green
	
	public FeedBack() {
		setDefaultCloseOperation(EXIT_ON_CLOSE);
		green.setBackground(Color.GREEN);
		red.setBackground(Color.RED);
		setLayout(new BorderLayout());
		setBounds(200, 200, 300,200);
		setVisible(true);
		state = -1;
	}

	public void showGreen() {
		getContentPane().removeAll();
		getContentPane().add(green, BorderLayout.CENTER);
		getContentPane().doLayout();
		update(getGraphics());
		state = 0;
	}
	
	public void showRed() {
		getContentPane().removeAll();
		getContentPane().add(red, BorderLayout.CENTER);
		getContentPane().doLayout();
		update(getGraphics());
		state = 1;
	}

}
