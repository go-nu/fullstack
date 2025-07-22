package com.example.kkwhw

import android.graphics.Color
import android.os.Bundle
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import com.example.kkwhw.databinding.ActivityChartBinding
import com.example.kkwhw.viewmodel.ExpenseViewModel
import com.github.mikephil.charting.charts.PieChart
import com.github.mikephil.charting.components.Legend
import com.github.mikephil.charting.data.PieData
import com.github.mikephil.charting.data.PieDataSet
import com.github.mikephil.charting.data.PieEntry

class ChartActivity : AppCompatActivity() {
    private lateinit var binding: ActivityChartBinding
    private val viewModel: ExpenseViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityChartBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val chart: PieChart = binding.pieChart
        viewModel.allExpenses.observe(this) { expenses ->
            val categoryTotals = expenses.groupBy { it.category }.mapValues { entry ->
                entry.value.sumOf { it.amount }
            }
            val entries = categoryTotals.map { PieEntry(it.value.toFloat(), it.key) }

            val dataSet = PieDataSet(entries, "카테고리별 지출")
            dataSet.colors = listOf(
                Color.RED, Color.BLUE, Color.GREEN, Color.MAGENTA, Color.CYAN, Color.YELLOW
            )
            val data = PieData(dataSet)

            chart.data = data
            chart.description.isEnabled = false
            chart.setUsePercentValues(true)
            chart.setEntryLabelColor(Color.BLACK)
            chart.setDrawEntryLabels(true)
            chart.legend.orientation = Legend.LegendOrientation.VERTICAL
            chart.legend.horizontalAlignment = Legend.LegendHorizontalAlignment.RIGHT
            chart.legend.verticalAlignment = Legend.LegendVerticalAlignment.CENTER
            chart.invalidate()
        }
    }
}
