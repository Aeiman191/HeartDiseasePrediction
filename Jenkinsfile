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
<<<<<<< HEAD
        }
=======
        }git 
>>>>>>> c60cc9dc1cdac3a413bde7fbb29ca3ef066dac39
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
        stage('Clean up local image') {
            steps {
                bat "docker rmi $registry:$BUILD_NUMBER"
            }
        }
    }
}
