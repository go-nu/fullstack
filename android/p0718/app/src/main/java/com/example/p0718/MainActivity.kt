package com.example.p0718

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import java.time.LocalDate

class MainActivity : AppCompatActivity() {

    private lateinit var viewModel: ExpenseViewModel
    private lateinit var adapter: ExpenseAdapter
    private lateinit var rvExpenses: RecyclerView // <<< RecyclerView 멤버 변수로 선언

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewModel = ViewModelProvider(this)[ExpenseViewModel::class.java]

        // RecyclerView 초기화 및 LayoutManager 설정
        rvExpenses = findViewById(R.id.rv_expenses) // <<< ID로 RecyclerView 인스턴스 가져오기
        rvExpenses.layoutManager = LinearLayoutManager(this) // <<< LayoutManager 설정

        adapter = ExpenseAdapter()
        rvExpenses.adapter = adapter // <<< RecyclerView에 어댑터 연결

        val today = LocalDate.now().toString()
        viewModel.getTodayTotal(today).observe(this) {
            findViewById<TextView>(R.id.tv_today_total).text = "오늘 지출: ${it ?: 0}원"
        }

        viewModel.allExpenses.observe(this) { expenses ->
            Log.d("MainActivity", "지출 리스트 수: ${expenses.size}")
            if (expenses.isEmpty()) { // <<< 데이터가 비어있을 경우 로그 추가
                Log.d("MainActivity", "표시할 지출 내역이 없습니다.")
            }
            adapter.submitList(expenses)
        }

        findViewById<Button>(R.id.btn_add_expense).setOnClickListener {
            startActivity(Intent(this, AddExpenseActivity::class.java))
        }
    }
}