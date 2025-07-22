package com.example.kkwhw

import android.os.Bundle
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.kkwhw.databinding.ActivityMonthlyStatsBinding
import com.example.kkwhw.model.MonthlyTotal
import com.example.kkwhw.viewmodel.ExpenseViewModel

class MonthlyStatsActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMonthlyStatsBinding
    private val viewModel: ExpenseViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMonthlyStatsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val adapter = MonthlyStatsAdapter()
        binding.recyclerMonthly.layoutManager = LinearLayoutManager(this)
        binding.recyclerMonthly.adapter = adapter

        viewModel.allExpenses.observe(this) { expenses ->
            val grouped = expenses.groupBy { it.date.substring(0, 7) } // yyyy-MM
            val monthlyTotals = grouped.map { (month, list) ->
                MonthlyTotal(month, list.sumOf { it.amount })
            }.sortedByDescending { it.date }
            adapter.submitList(monthlyTotals)
        }
    }
}