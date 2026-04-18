#include <QCoreApplication>
#include <QtLlama/Session.h>
#include <QtLlama/Engine.h>
#include <QDebug>


int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);


    if(argc < 2){
        qCritical().noquote() << "Usage: helloWorld </path/to/model.gguf>";
        return 0;
    }
    const QString MODEL_PATH = argv[1];

    if(!QFile::exists(MODEL_PATH)){
        qCritical().noquote() << "File does not exist at: " << MODEL_PATH;
        return 0;
    }

    QtLlama::Session session;
    session.setConfig({.modelPath = MODEL_PATH});
    session.initialize(new QtLlama::Engine());


    QString systemPrompt = "You are a helpful assistant, your answer should not exceede 2 sentences:\n";
    
    QList<QString> questions = {
        "What is the capital of France?",
        "How many planets are in the solar system?",
        "What is the chemical symbol for gold?"
    };



    auto askNext = [&](){
        static size_t i = 0;
        if(i >= questions.size()) return;
        const QString q =  questions.at(i);

        QString prompt = q;         
        session.generate(prompt, systemPrompt, i);
        i++;
    };

    QObject::connect(&session, &QtLlama::Session::responseReady, [&](const QString& response, size_t i){
        qDebug().noquote() << "Q: " << questions.at(i) << "\nA: " << response << "\n";
        askNext();
    });

    QObject::connect(&session, &QtLlama::Session::statusChanged, [&](auto status){
        switch (status) {
            case QtLlama::Status::Loading: {
                qInfo().noquote() << "Model is being loaded!\n";
            }

            case QtLlama::Status::Ready: {
                qInfo().noquote() << "Model loaded successfully! Starting Q&A...\n";
                askNext();
                break;
            }

            case QtLlama::Status::Error: {
                qCritical().noquote() << "Error:  " << session.statusText() << '\n';
                break;
            }
        }
    });


    session.loadModel();


    return app.exec();
}
#include "main.moc"

