#include <QtLlama/Session.h>
#include <QDebug>

namespace QtLlama {

Session::Session(QObject *parent) : QObject(parent) {}

Session::~Session() {
    if (mWorkerThread) {
        mWorkerThread->quit();
        mWorkerThread->wait();
    }
}

void Session::initialize(IEngine* engine) {
    if (mWorkerThread) return;
    
    qRegisterMetaType<QList<QtLlama::Message>>("QList<QtLlama::Message>");
    qRegisterMetaType<QtLlama::Message>("QtLlama::Message");
    
    Q_ASSERT_X(engine != nullptr, "Session::initialize", "engine must not be null");
    Q_ASSERT_X(mEngine == nullptr, "Session::initialize", "initialize() called twice");

    mEngine = engine;
    mWorkerThread = new QThread(this);

    mEngine->setParent(nullptr);
    mEngine->moveToThread(mWorkerThread);
    mEngine->setConfig(mConfig);

    connect(mEngine, &IEngine::tokenGenerated, this, [this](const QString& chunk){
        emit textGenerated(chunk, mCurrentSessionId);
    });

    connect(mEngine, &IEngine::isGeneratingChanged, this, [this](bool busy){
        if (mIsGenerating != busy) {
            mIsGenerating = busy;
            emit isGeneratingChanged(busy);
        }
    });

    connect(mEngine, &IEngine::modelStatusChanged, this, [this](Status s){
        if (mStatus != s) {
            mStatus = s;
            emit statusChanged(mStatus);
        }
    });

    connect(mEngine, &IEngine::errorOccurred, this, [this](const QString& msg){
        emit errorOccurred(msg);
    });


    connect(mEngine, &IEngine::responseReady, this, [this](const QString& full){
        emit responseReady(full, mCurrentSessionId);
    });

    connect(mWorkerThread, &QThread::finished, mEngine, &QObject::deleteLater);
    mWorkerThread->start();
}

void Session::setConfig(const Config &config) {
    *mConfig = config; 
    if (mEngine)
        QMetaObject::invokeMethod(mEngine, "setConfig", Q_ARG(QSharedPointer<Config>, mConfig));
}

void Session::generate(const QList<QtLlama::Message>& messages, int sessionId) {
    if (mStatus != Status::Ready) return;
    mCurrentSessionId = sessionId;
    QMetaObject::invokeMethod(mEngine, "generate", Q_ARG(QList<QtLlama::Message>, messages));
}


void Session::generate(const QString& userMessage, int sessionId) {
    generate({ { Role::User, userMessage } }, sessionId);
}

void Session::generate(const QString& userMessage, const QString& systemPrompt, int sessionId) {
    generate({
        { Role::System, systemPrompt },
        { Role::User,   userMessage  }
    }, sessionId);
}

void Session::stop() {
    if (mEngine) QMetaObject::invokeMethod(mEngine, "stop");
}

void Session::loadModel() {
    if (!mConfig || mConfig->modelPath.isEmpty()) return;
    QMetaObject::invokeMethod(mEngine, "loadModel");
}

void Session::unloadModel() {
    if (mEngine) QMetaObject::invokeMethod(mEngine, "unloadModel");
}

QString Session::statusText() const {
    switch (mStatus) {
        case Status::Idle:    return tr("No Model Loaded");
        case Status::Loading: return tr("Loading Model...");
        case Status::Ready:   return tr("Ready");
        case Status::Error:   return tr("Model Error");
        default:              return tr("Unknown");
    }
}

} // namespace QtLlama
