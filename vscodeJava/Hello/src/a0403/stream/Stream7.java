package a0403.stream;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class Stream7 {
    public static void main(String[] args) {    
        List<Integer> list = (List) Arrays.asList(10, 20, 30, 40, 50);
        Iterator<Integer> iter = list.iterator();
        while(iter.hasNext()) {
            Integer value = iter.next();
            System.out.println(value + " ");
        }


    }
}