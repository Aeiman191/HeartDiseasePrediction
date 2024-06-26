pipeline {
    environment {
        registry = "aeiman/heartdiseaseprediction" // Your Docker Hub account/repository
        registryCredential = 'docker-hub-credentials' // Jenkins credential ID
        dockerImage = ''
    }
    agent any
    stages {
        stage('Get Dockerfile from GitHub') {
            steps {
                git branch: 'main', url: 'https://github.com/Aeiman191/HeartDiseasePrediction.git' // Your GitHub repository
            }
        }
        stage('Build Docker image') {
            steps {
                script {
                    dockerImage = docker.build(registry + ":$BUILD_NUMBER")
                }
            }
        }
        stage('Push Docker image to Docker Hub') {
            steps {
                script {
                    docker.withRegistry('', registryCredential) {
                        dockerImage.push()
                    }
                }
            }
        }
    }
    post {
        success {
            emailext (
                subject: "Docker Image Build Successful",
                body: "The Docker image build was successful.",
                recipientProviders: [[$class: 'CulpritsRecipientProvider']],
                attachLog: true
            )
        }
    }
}
