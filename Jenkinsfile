pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                // Pull latest changes from main branch
                git branch: 'main', url: 'https://github.com/Aeiman191/HeartDiseasePrediction.git'
            }
        }
        stage('Build and Push Docker Image') {
            steps {
                // Build and push Docker image to Docker Hub
                script {
                    // Example Docker commands for Windows
                    bat 'docker build -t aeiman/heartdiseaseprediction .'
                    bat 'docker login -u aeiman -p khan12345'
                    bat 'docker tag aeiman/heartdiseaseprediction aeiman/heartdiseaseprediction:latest'
                    bat 'docker push aeiman/heartdiseaseprediction:latest'
                }
            }
        }
        
        stage('Email Notification') {
            steps {
                emailext body: 'Docker image built and pushed successfully to Docker Hub.',
                         subject: 'CI/CD Pipeline Notification',
                         to: 'aimteiyaz191@gmail.com'
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
    }
}
