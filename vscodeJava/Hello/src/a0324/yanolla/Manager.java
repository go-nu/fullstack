package a0324.yanolla;

import java.util.ArrayList;

public class Manager {
    private ArrayList<Accommodation> accommodations;
    private ArrayList<Accommodation> reserved;

    public Manager() {
        accommodations = new ArrayList<>();
        reserved = new ArrayList<>();

        accommodations.add(new Accommodation("Hotel A", "Seoul", 100.0));
        accommodations.add(new Accommodation("Hotel B", "Busan", 80.0));
        accommodations.add(new Accommodation("Hotel C", "Jeju", 120.0));
    }

    public void bookableList() {
        int count = 0;
        for (Accommodation bookableAccommodation : accommodations) {
            if (!bookableAccommodation.isAvailable()) {
                count++;
            } else {
                System.out.println(bookableAccommodation);
            }
        }
        
        if (count == accommodations.size()) {
            System.out.println("예약가능한 숙소가 없습니다.");
        }
    }

    public boolean reserveAccommodation(String reserveAccommodationName) {
        for (Accommodation reserveAccommodation : accommodations) {
            if(reserveAccommodation.getName().equalsIgnoreCase(reserveAccommodationName)
            && reserveAccommodation.isAvailable()) {
                reserved.add(reserveAccommodation);
                reserveAccommodation.reservation();
                
                return true;
            }
        }
        return false;
    }

    public void checkReserved() {
        for(Accommodation reservedAccommodation : reserved) {
            System.out.println(reservedAccommodation);
        }
    }

    public void addAccommodation(String addName, String addLocation, double addPrice) {
        accommodations.add(new Accommodation(addName, addLocation, addPrice));
    }


}
