package a0324.yanolla;

import java.util.ArrayList;
import java.util.Scanner;

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

    public void delAccommodation(String delName) {
        for(Accommodation delAccommodation : accommodations) {
            if(delAccommodation.getName().equalsIgnoreCase(delName)) {
                if(delAccommodation.isAvailable()) {
                    accommodations.remove(delAccommodation);
                    // return true;
                    System.out.println("삭제 완료.");
                    break;
                } else System.out.println("예약된 숙소입니다.");
            } else {
                System.out.println("숙소 이름이 잘못되었습니다.");
                break;
            }
        }
        // return false;
    }

    public void fixInfo(String fixName) {
        int i = 0;
        int index = -1;
        boolean flag = true;
        Scanner s = new Scanner(System.in);
        Accommodation fixA = new Accommodation();
        for(Accommodation fA : accommodations) {
            i++;
            if(fA.getName().equalsIgnoreCase(fixName)) {
                index = i - 1;
                fixA = fA;
                break;
            }
        }
        if(index != -1) {
            System.out.print("1. 숙소명 | 2. 숙소 위치 | 3. 숙소 가격 \n 수정 항목> ");
            int select = s.nextInt();
            s.nextLine();
            while (flag) {
                switch (select) {
                    case 1:
                        System.out.print("수정할 숙소명 : ");
                        fixA.setName(s.nextLine());
                        accommodations.set(index, fixA);
                        flag = false;
                        break;
                    case 2:
                        System.out.print("수정할 숙소 위치 : ");
                        fixA.setLocation(s.nextLine());
                        accommodations.set(index, fixA);
                        flag = false;
                        break;
                    case 3:
                        System.out.print("수정할 숙소 가격: ");
                        fixA.setPrice(s.nextDouble());
                        accommodations.set(index, fixA);
                        flag = false;
                        break;
                    default:
                        System.out.println("올바른 번호를 입력하세요");
                        break;
                }
            }
        } else System.out.println("찾는 숙소가 없습니다.");
        s.close();
    }

    public void checkAccommodation(String checkName) {
        for(Accommodation ca : accommodations) {
            if(ca.getName().equalsIgnoreCase(checkName)) {
                System.out.println(ca.toString());
            }
        }
    }


}
