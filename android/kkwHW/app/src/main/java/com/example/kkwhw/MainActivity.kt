package com.example.kkwhw

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.kkwhw.databinding.ActivityMainBinding
import com.example.kkwhw.viewmodel.ExpenseViewModel
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val viewModel: ExpenseViewModel by viewModels()
    private val adapter = ExpenseListAdapter { expense ->
        viewModel.delete(expense)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.recyclerView.layoutManager = LinearLayoutManager(this)
        binding.recyclerView.adapter = adapter

        val today = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(Date())
        viewModel.getTotalAmountByDate(today).observe(this) {
            binding.textTotal.text = "오늘 지출: ${it ?: 0}원"
        }

        viewModel.allExpenses.observe(this) {
            adapter.submitList(it)
        }

        val categories = listOf("전체", "식비", "교통비", "쇼핑", "기타")
        binding.spinnerFilter.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, categories)
        binding.spinnerFilter.setSelection(0)

        binding.spinnerFilter.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selected = categories[position]
                if (selected == "전체") {
                    viewModel.allExpenses.observe(this@MainActivity) {
                        adapter.submitList(it)
                    }
                } else {
                    viewModel.getExpensesByCategory(selected).observe(this@MainActivity) {
                        adapter.submitList(it)
                    }
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        binding.buttonAdd.setOnClickListener {
            startActivity(Intent(this, AddExpenseActivity::class.java))
        }

        binding.buttonStats.setOnClickListener {
            startActivity(Intent(this, MonthlyStatsActivity::class.java))
        }

        binding.buttonChart.setOnClickListener {
            startActivity(Intent(this, ChartActivity::class.java))
        }
    }
}
