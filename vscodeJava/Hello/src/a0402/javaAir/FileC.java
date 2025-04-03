package a0402.javaAir;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Map;

// 디렉토리(폴더가 없는 경우 폴더 생성)

public class FileC {
    private FlightManager fm = new FlightManager();

    public void ticketSaveFile(Map<String, Flight> reservationMap, String name) {
        try {
            File file = new File("e:\\ticket\\ticket.txt");
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(file));
            // BufferedWriter를 사용하여 file에 데이터를 쓸 준비
            // FileWriter는 기본적으로 기존 파일을 덮어쓴다.
            if(file.isFile() && file.canWrite()) {
                // .canWrite() - 쓰기 권한이 있는지 true / false
                bufferedWriter.write(fm.ticketPrint(reservationMap, name)); // 티켓정보를 파일에 쓰기
                bufferedWriter.flush(); // 버퍼에 있는 내용을 파일에 저장
                System.out.println("저장 완료");
                bufferedWriter.close();
            }
        } catch (IOException e) {
            System.out.println("파일 저장 실패");
        }    
    }

    public void upload() {
        try {
            File file = new File("e:\\ticket\\schedule1.txt");
            BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
            // BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(new FileReader(file)));
            String line;
            System.out.println("========================================");
            while ((line = bufferedReader.readLine()) != null) {
                System.out.println(line);
                String[] flight = line.split("/");
                FlightManager.getFlights().add(new Flight(flight[0], flight[1], Integer.parseInt(flight[2]), Boolean.parseBoolean(flight[3])));

            }
        } catch (FileNotFoundException e) {
            System.out.println("schedule.txt 파일을 찾을 수 없습니다.");
        } catch (IOException e) {
            System.out.println("파일 읽기 실패");
        }
    }

}
