# PR: feat(di): add DI container, factories and UML docs; extend builder and fix retrieval strategies

Kurzbeschreibung

- Fügt einen einfachen DI-Container (src/di_container.py) hinzu mit Singleton/Transient/Scoped Support.
- Implementiert Factories (src/factories.py) für LLMs, Vector- und Graph-Stores sowie Retrieval-Strategien.
- Erweitert den RAG-System-Builder (src/config/builders.py) um with_temperature und with_retrieval_k.
- Behebt einen Fehler in retrieval_strategies (Fehlende datetime-Importe).
- Fügt UML-Quellen und eine Implementierungsstrategie-Dokumentation in docs/uml/ und docs/IMPLEMENTATION_STRATEGY.md hinzu.

Änderungen (Auswahl)

- src/di_container.py (neu/erweitert)
- src/factories.py (neu/erweitert)
- src/config/builders.py (erweitert)
- src/strategies/retrieval_strategies.py (kleine Fixes)
- docs/uml/*.puml (neu)
- docs/IMPLEMENTATION_STRATEGY.md (neu)
- tests/test_di_factories.py (neu)

Tests

- Lokale Test-Suite: 17 passed
  - pytest -q

Wie testen / reviewen

1. Lokale Tests ausführen:
   - python -m pytest -q
2. Quick smoke:
   - pytest tests/test_di_factories.py::test_factories_registration_and_create -q
   - pytest tests/test_patterns.py::test_dependency_injection -q

UML Rendering (optional)

Zum Erzeugen von PNG/SVG aus den .puml Dateien:

- Mit plantuml.jar:
  - curl -L -o plantuml.jar https://repo1.maven.org/maven2/net/sourceforge/plantuml/plantuml/1.2023.10/plantuml-1.2023.10.jar
  - java -jar plantuml.jar -tpng docs/uml/*.puml -o docs/uml

Nächste Schritte / Vorschlag

- Review und Merge in main nach Code-Review.
- Option: PlantUML rendern und Bilder in docs/uml/ ablegen.
- Option: CI-Workflow (GitHub Actions) sicherstellen, dass pytest automatisch läuft.

Reviewer Vorschlag

- @sijadev (Primary)


