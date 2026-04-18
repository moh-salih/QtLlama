#pragma once
#include <llama.h>
#include <QMutex>
#include <QDebug>
#include <QMetaType>

#include <QtLlama/Types.h>

namespace QtLlama {
   inline void ensureBackendInit() {
        static QBasicMutex mutex;
        static bool initialized = false;
        QMutexLocker lock(&mutex);
        if (!initialized) {
            llama_backend_init();
            initialized = true;
        }

        llama_log_set([](enum ggml_log_level level, const char* text, void*) {
            switch (level) {
                case GGML_LOG_LEVEL_ERROR: qCritical() << "[llama.cpp]" << text; break;
                case GGML_LOG_LEVEL_WARN:  qWarning()  << "[llama.cpp]" << text; break;
                default: break; // suppress info/debug by default
            }
        }, nullptr);
    }



   inline void ensureMetaTypesRegistered() {
        static QBasicMutex mutex;
        static bool registered = false;
        QMutexLocker lock(&mutex);
        if (registered) return;
        qRegisterMetaType<std::vector<float>>("std::vector<float>");
        qRegisterMetaType<QtLlama::Message>("Message");
        qRegisterMetaType<QList<QtLlama::Message>>("QList<Message>");
        registered = true;
    }
}
