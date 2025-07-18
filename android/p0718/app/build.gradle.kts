plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    id("kotlin-kapt") // Room에서 필요
}

android {
    namespace = "com.example.p0718"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.p0718"
        minSdk = 30
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    buildFeatures {
        viewBinding = true
    } // findById() 레이아웃 접근을 viewBinding에서 대체
    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.lifecycle.viewmodel.android)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    // Room (from version catalog)
    implementation(libs.androidx.room.runtime)
    // RoomDatabase - DAO, ENTITY 기능 제공
    implementation(libs.androidx.room.ktx)
    // Room의 kotlin 확장기능
    kapt(libs.androidx.room.compiler)
    // @Entity 등 어노테이션 자동 코드 생성
    implementation(libs.androidx.lifecycle.viewmodel.ktx)
    // viewModel 사용하기 위함.
}