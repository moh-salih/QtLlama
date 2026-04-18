#include <QMetaType>
#include <QtLlama/Embedder.h>

namespace QtLlama {

Embedder::Embedder(QObject *parent) : QObject(parent) {}

Embedder::~Embedder() {
    if (mWorkerThread) {
        mWorkerThread->quit();
        mWorkerThread->wait();
    }
}

void Embedder::initialize(IEmbedder* engine) {
    if (mWorkerThread) return;

    qRegisterMetaType<std::vector<float>>("std::vector<float>");

    mEngine = engine;
    mWorkerThread = new QThread(this);
    mEngine->setParent(nullptr);
    mEngine->moveToThread(mWorkerThread);
    mEngine->setConfig(mConfig);

    connect(mEngine, &IEmbedder::vectorReady, this, &Embedder::embeddingReady);

    connect(mEngine, &IEmbedder::isGeneratingChanged, this, [this](bool busy){
        if (mIsGenerating != busy) {
            mIsGenerating = busy;
            emit isGeneratingChanged(busy);
        }
    });

    connect(mEngine, &IEmbedder::modelStatusChanged, this, [this](Status s){
        if (mStatus != s) {
            mStatus = s;
            emit statusChanged(mStatus);
        }
    });

    connect(mEngine, &IEmbedder::errorOccurred, this, &Embedder::errorOccurred);

    connect(mWorkerThread, &QThread::finished, mEngine, &QObject::deleteLater);
    mWorkerThread->start();
}

void Embedder::setConfig(const EmbedConfig &config) {
    *mConfig = config;
    if (mEngine)
        QMetaObject::invokeMethod(mEngine, "setConfig", Q_ARG(QSharedPointer<EmbedConfig>, mConfig));
}

void Embedder::generateEmbedding(const QString& text, int chunkIndex) {
    if (mStatus != Status::Ready) return;
    QMetaObject::invokeMethod(mEngine, "generateEmbedding", Q_ARG(QString, text), Q_ARG(int, chunkIndex));
}

void Embedder::loadModel() {
    if (!mConfig || mConfig->modelPath.isEmpty()) return;
    QMetaObject::invokeMethod(mEngine, "loadModel");
}

void Embedder::unloadModel() {
    QMetaObject::invokeMethod(mEngine, "unloadModel");
}

void Embedder::stop() {
    if (mEngine) QMetaObject::invokeMethod(mEngine, "stop");
}

QString Embedder::statusText() const {
    switch (mStatus) {
    case Status::Idle:    return tr("No Embedding Model");
    case Status::Loading: return tr("Loading Vector Engine...");
    case Status::Ready:   return tr("Ready");
    case Status::Error:   return tr("Vector Engine Error");
    default:              return tr("Unknown");
    }
}

} // namespace QtLlama
