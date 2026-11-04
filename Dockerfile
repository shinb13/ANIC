FROM agenthack/base:1.1.1

USER root
WORKDIR /app
COPY agentcc_solver.py ./
COPY src/ ./src/
USER agent

LABEL org.agenthack.team_id="team-001"
LABEL org.agenthack.version="0.1.0"

EXPOSE 8080
CMD ["python", "-m", "src"]
