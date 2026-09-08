"""算子调度器（被测对象）：DAG 依赖 + 资源约束 + 优先级的贪心调度。

模拟 AI 芯片软件栈的算子调度系统：
  - 每个任务有 deps（依赖）、cost（资源占用）、priority（越大越优先）
  - 每个时间槽内被调度任务的 cost 之和不得超过 resource_budget
  - 违反约束 / 死锁场景抛出明确异常（详见 tests/test_scheduler/）
"""
from dataclasses import dataclass, field


class SchedulerError(ValueError):
    pass


class CycleError(SchedulerError):
    pass


class MissingDependencyError(SchedulerError):
    pass


class UnschedulableError(SchedulerError):
    pass


@dataclass
class Task:
    id: str
    cost: int = 1
    priority: int = 0
    deps: tuple[str, ...] = ()


@dataclass
class Schedule:
    """调度结果：slot -> 任务 id 列表。"""

    slots: list[list[str]] = field(default_factory=list)

    def order(self) -> list[str]:
        return [tid for slot in self.slots for tid in slot]

    def makespan(self) -> int:
        return len(self.slots)


def schedule(tasks: list[Task], resource_budget: int) -> Schedule:
    """贪心列表调度：每槽从就绪任务中按 (priority 降序, id 升序) 尽量装入。"""
    if resource_budget <= 0:
        raise SchedulerError(f"资源预算必须为正, got {resource_budget}")
    by_id: dict[str, Task] = {}
    for t in tasks:
        if t.id in by_id:
            raise SchedulerError(f"任务 id 重复: {t.id}")
        if t.cost <= 0:
            raise SchedulerError(f"任务 {t.id} cost 必须为正, got {t.cost}")
        by_id[t.id] = t
    for t in tasks:
        for dep in t.deps:
            if dep not in by_id:
                raise MissingDependencyError(f"任务 {t.id} 依赖不存在的任务 {dep}")
            if dep == t.id:
                raise CycleError(f"任务 {t.id} 依赖自身")

    scheduled: set[str] = set()
    slots: list[list[str]] = []
    # 依赖链是否成环：全部调度完前若某槽无任何任务可入槽则判定死锁
    while len(scheduled) < len(tasks):
        ready = [
            t for t in tasks
            if t.id not in scheduled and all(d in scheduled for d in t.deps)
        ]
        ready.sort(key=lambda t: (-t.priority, t.id))
        slot, used = [], 0
        for t in ready:
            if used + t.cost <= resource_budget:
                slot.append(t.id)
                used += t.cost
        if not slot:
            # 有就绪任务却装不下（单任务超预算），或全部剩余任务互相等待（环）
            if ready and all(t.cost > resource_budget for t in ready):
                raise UnschedulableError(
                    f"任务 {[t.id for t in ready]} 的 cost 超过预算 {resource_budget}"
                )
            raise CycleError("剩余任务存在循环依赖，无法调度")
        scheduled.update(slot)
        slots.append(slot)
    return Schedule(slots=slots)
