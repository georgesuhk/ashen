!> Trace relativistic guiding centres through the fields of an existing JOREK run.
!>
!> A configurable version of ex7_jorek, for ashen's `ptrace` (a case's
!> ptrace_exe pointing at the built binary). Built only on the HPC -- ashen
!> is developed without a Fortran compiler. Everything ex7_jorek hard-codes
!> is read from the &ptrace namelist in ptrace_settings.nml, next to the
!> binary's working folder. ashen writes that file into the trace folder
!> from the case's ptrace_<name> keys in cases.toml (ptrace_dt,
!> ptrace_initialiser, ...) over its defaults -- every setting, so it is also
!> the record of what the trace ran with. (Run by hand: write one yourself.)
!> The values below are this program's own defaults, the same as ashen's
!> (ashen.particle_programs.PTRACE_DEFAULTS):
!>
!>   &ptrace
!>     field_mode    = 'evolving'  ! 'static': jorek_restart.h5 only, frozen
!>                                 ! 'evolving': jorek<i>.h5, jorek<i+1>.h5, ...
!>                                 !   interpolated linearly in time
!>     restart_index = 0           ! evolving: index of the first restart file
!>     hold_last_field = .false.   ! evolving: past the last restart, keep its
!>                                 !   field frozen (.true.) or stop there
!>                                 !   (.false.), with every output written
!>     t_span        = 0.d0        ! [s] traced, from the first restart's time;
!>                                 !   0 = until the last linked restart's time,
!>                                 !   i.e. ptrace_end_step under ashen
!>     dt            = 1.d-10      ! [s] RK4 step
!>     diag_step     = 1.d-8       ! [s] between diagnostics writes
!>     n_snapshots   = 100         ! about how many part_restart_s<step>_t<time>.h5
!>                                 !   files over the trace: one every (traced
!>                                 !   time)/n_snapshots, rounded up to 1, 2 or
!>                                 !   5 x 10^k s, so 40 to 100 of them at round
!>                                 !   times. 0 = only the final part_restart.h5
!>     snapshot_step = 0.d0        ! [s] between them, if > 0: used instead of
!>                                 !   n_snapshots
!>     stop_when_stalled = .false. ! stop before the end once the markers have
!>                                 !   all but stopped leaving, see below
!>     stall_rate_fraction = 5.d-2 ! ...: the loss rate, as a fraction of its peak
!>     stall_min_lost = 1.d-1      ! ...: not before this fraction of the markers
!>                                 !   has left,
!>     stall_min_time = 5.d-1      ! ...or this fraction of the trace has passed
!>     stall_window  = 0.d0        ! [s] the loss rate is measured over; 0 = a
!>                                 !   tenth of the trace
!>     mass          = 5.48579909065d-4  ! [amu]
!>     initialiser   = 'markers'   ! how the markers are placed, see below
!>     n_markers     =             ! no default: how many markers
!>     R0 =   Z0 =   phi0 = 0.           ! [m], [m], [rad]; R0, Z0: no default
!>     E_kin_eV =                        ! kinetic energy [eV]; no default
!>     cos_pitch =                       ! v_par/v, in -1..1; no default
!>     charge = -1                       ! [e]
!>   /
!>
!> Initialisers (initialiser = ...):
!>   'markers'            -- each marker k at (R0(k), Z0(k), phi0(k)) with its
!>                           own E_kin_eV(k), cos_pitch(k), charge(k)
!>   'current_pdf_simple' -- n_markers placed with a restart's toroidal
!>                           current density as their pdf: jorek_pdf.h5
!>                           (ashen links ptrace_pdf_step's restart as it),
!>                           else the first restart. All with E_kin_eV(1),
!>                           cos_pitch(1), charge(1); R0/Z0/phi0 are ignored.
!>                           The current is its n = 0 (axisymmetric) part.
!>                           Settings:
!>       pdf_n_sub = 4    ! each grid element split into n_sub x n_sub cells
!>       seed      = 1    ! random seed; same seed, same markers
!>       (pdf_n_phi is still accepted, and ignored: the n = 0 part is
!>       taken directly now, not averaged over toroidal planes.)
!>     Particles per unit R-Z area go as R*j_phi, i.e. as JOREK's zj
!>     (Delta* psi), taking only the current along the net plasma current --
!>     counter-current regions get none. Each cell's share is its zj times
!>     its area; within a cell a marker is uniform in (s, t), and phi is
!>     uniform in [0, 2 pi). Sampling is from that table, so it cannot hang
!>     the way re_gc's rejection sampling can.
!>
!> Stopping early. Two rules end the trace before its end time, with every
!> output written as at the end and a 'ptrace_gc: NOTE:' line saying why:
!>   - every marker is off the grid: nothing is left to push. Always on.
!>   - stop_when_stalled: the markers have all but stopped leaving. The loss
!>     rate -- markers that left per second, over the last stall_window --
!>     has fallen to stall_rate_fraction of the highest it has been. Only
!>     looked at once stall_min_lost of the markers has left or
!>     stall_min_time of the trace has passed, whichever comes first, and
!>     never while no marker has left at all. A marker has left when it is
!>     off the grid or, if ptrace_boundary.dat is there (ashen writes the
!>     plasma boundary before extend_bnd into it: a line with n, then n lines
!>     of R Z), has been outside that outline -- once out, it counts as left
!>     for good. This rule cannot tell a field that has saturated from one
!>     that has not started moving yet: the two gates are what keeps it from
!>     stopping a trace before its losses begin, so set them for the case.
!> Both are looked at only where the trace stops anyway (every diag_step
!> and snapshot), so switching them on changes no marker's path.
!>
!> Under ashen, keep restart_index = 0: ashen links the chosen restarts in
!> as a consecutive sequence from index 0 (the reader looks only i+1..i+20
!> past file i, at rst_file_ind_fmt(1)'s width), and links jorek_restart.h5
!> to the start restart for field_mode = 'static'. See ashen.ptracing.
!>
!> The start time is the first restart's own time, never set separately:
!> the field reader sets sim%time from the file, and a separately given
!> start time that disagreed with it would only be warned about.
!>
!> Differences from ex7_jorek:
!> - a particle that leaves the grid is flagged lost (i_elm <= 0) and no
!>   longer pushed; the run carries on with the others instead of `stop`
!> - each particle is pushed with its own local time (re_gc_current_density_
!>   initialisation's loop). ex7 advanced sim%time inside the particle loop,
!>   so with more than one particle every later one saw fields at the wrong time
!> - markers are shared round-robin between MPI ranks
!> - by default it runs from the first linked restart to the last one
!>   (ptrace_start_step to ptrace_end_step under ashen), static or evolving
!> - with hold_last_field = .false., a t_span > 0 reaching past the last
!>   restart stops at that restart's time, writes its outputs as usual and
!>   logs a 'ptrace_gc: NOTE:' line, where JOREK's field reader would
!>   MPI_Abort with nothing written
!>
!> Outputs: ptrace_diag.h5 (write_particle_diagnostics: energy, mu, psi_n,
!> p_phi, lost, theta, phi, R, Z per diag_step -- what `plot --diag
!> particle_exits` reads); every snapshot_step (see n_snapshots) from the start,
!> part_restart_s<step>_t<time>.h5 -- <step> the JOREK step (index_now) of
!> the restart closest in time, <time> the simulation time in seconds, e.g.
!> part_restart_s003200_t2.500000E-03.h5; and part_restart.h5 at the end.
!> All are write_simulation_hdf5 files, the ones `plot --diag particles` draws.
!>
!> Kept in ashen, not in JOREK (which ashen never modifies). To build, copy
!> it into a JOREK checkout's particles/examples/ -- the Makefile picks up
!> any program there by filename -- then `make ptrace_gc` with the same MODEL
!> as the run being traced.
!> Run with `mpirun -n N ./ptrace_gc < in_main`, next to ptrace_settings.nml.

program ptrace_gc

! Every import is named: particle_tracer re-exports everything it uses
! (phys_module's t_start among it), and a local name clashing with any of
! that would not compile. find_RZ is the external grids/grid_utils/find_RZ.f90.
use particle_tracer,          only: sim, events, particle_sim, particle_gc_relativistic, &
                                    event, with, next_event_at, stop_action
use mod_particle_io,          only: write_simulation_hdf5
use mod_particle_diagnostics, only: write_particle_diagnostics
use mod_fields_linear,        only: read_jorek_fields_interp_linear
use mod_gc_relativistic,      only: runge_kutta_fixed_dt_gc_push_jorek, &
                                    relativistic_gc_momenta_from_E_cospitch
use phys_module,              only: central_mass, central_density
use constants,                only: ATOMIC_MASS_UNIT, EL_CHG, SPEED_OF_LIGHT, MU_ZERO
use mpi

implicit none

integer, parameter :: MAX_MARKERS = 100000
!> The one file the &ptrace namelist is read from.
character(len=*), parameter :: SETTINGS_FILE = 'ptrace_settings.nml'

! --- &ptrace namelist ---
character(len=16) :: field_mode = 'evolving'
integer           :: restart_index = 0
logical           :: hold_last_field = .false.
real*8            :: t_span = 0.d0, dt = 1.d-10, diag_step = 1.d-8, snapshot_step = 0.d0
integer           :: n_snapshots = 100
logical           :: stop_when_stalled = .false.
real*8            :: stall_rate_fraction = 5.d-2, stall_min_lost = 1.d-1, stall_min_time = 5.d-1
real*8            :: stall_window = 0.d0
real*8            :: mass = 5.48579909065d-4
integer           :: n_markers = 0
real*8            :: R0(MAX_MARKERS) = 0.d0, Z0(MAX_MARKERS) = 0.d0, phi0(MAX_MARKERS) = 0.d0
! cos_pitch has no default: 2 is outside -1..1, so leaving it unset is caught below.
real*8            :: E_kin_eV(MAX_MARKERS) = 0.d0, cos_pitch(MAX_MARKERS) = 2.d0
integer           :: charge(MAX_MARKERS) = -1
character(len=32) :: initialiser = 'markers'
integer           :: pdf_n_sub = 4, pdf_n_phi = 16, seed = 1  ! pdf_n_phi: ignored, kept readable
namelist /ptrace/ field_mode, restart_index, hold_last_field, t_span, dt, diag_step, &
                 snapshot_step, mass, n_markers, R0, Z0, phi0, E_kin_eV, cos_pitch, charge, &
                 initialiser, pdf_n_sub, pdf_n_phi, seed, n_snapshots, &
                 stop_when_stalled, stall_rate_fraction, stall_min_lost, stall_min_time, &
                 stall_window

!> Two times closer than this [s] are the same time (mod_event's TICK).
real*8, parameter :: SNAP_TICK = 1.d-12
!> The outline a marker has left the plasma by crossing, if this file is there.
character(len=*), parameter :: BOUNDARY_FILE = 'ptrace_boundary.dat'
!> The loss rate is sampled this many times per stall_window.
integer, parameter :: SAMPLES_PER_WINDOW = 4

! sim and events come from particle_tracer
type(event)                      :: field_reader
type(write_particle_diagnostics) :: diag
type(particle_gc_relativistic)   :: marker
real*8                           :: t_start, t_stop, target_time, rest_energy_eV, t_snap
integer                          :: u, io, ierr, k, j, n_local, ifail, n_lost, n_snap
integer                          :: my_rank, n_ranks
logical                          :: exists
! Stopping early: the boundary outline, which local markers have left, and
! the count of those that have at each sample time
integer                          :: n_bnd, n_left, n_samples
real*8,  allocatable             :: bnd_R(:), bnd_Z(:), sample_t(:)
integer, allocatable             :: sample_n(:)
logical, allocatable             :: has_left(:)
real*8                           :: t_check, max_loss_rate
! The restarts this run reads: their JOREK step (index_now) and time [s]
integer                          :: n_rst
integer, allocatable             :: rst_steps(:)
real*8,  allocatable             :: rst_times(:)

call sim%initialize(num_groups=1)
! Asked of MPI directly: particle_sim's own rank fields differ between JOREK versions.
call MPI_COMM_RANK(MPI_COMM_WORLD, my_rank, ierr)
call MPI_COMM_SIZE(MPI_COMM_WORLD, n_ranks, ierr)

! --- parameters: read on rank 0, broadcast ---
if (my_rank .eq. 0) then
  inquire(file=SETTINGS_FILE, exist=exists)
  if (.not. exists) then
    write(*,*) 'ERROR: ptrace_gc: no ', SETTINGS_FILE, ' here -- nothing says how many ', &
      'markers or where. ashen writes it from the case''s ptrace_<name> keys in ', &
      'cases.toml (ptrace_n_markers, ptrace_E_kin_eV, ...); ptrace_params.nml and ', &
      'ptrace_overrides.nml are no longer read'
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  open(newunit=u, file=SETTINGS_FILE, status='old', action='read', iostat=io)
  if (io .eq. 0) read(u, nml=ptrace, iostat=io)
  close(u)
  if (io .ne. 0) then
    write(*,*) 'ERROR: ptrace_gc: cannot read the &ptrace namelist in ', SETTINGS_FILE
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  write(*,*) 'ptrace_gc: settings from ', SETTINGS_FILE
  if (t_span .lt. 0.d0) then
    write(*,*) 'ERROR: ptrace_gc: t_span must be >= 0 (0 = until the last restart), got ', t_span
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  if (n_markers .lt. 1 .or. n_markers .gt. MAX_MARKERS) then
    write(*,*) 'ERROR: ptrace_gc: n_markers must be in 1..', MAX_MARKERS, ', got ', n_markers
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  if (snapshot_step .lt. 0.d0) then
    write(*,*) 'ERROR: ptrace_gc: snapshot_step must be >= 0, got ', snapshot_step
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  if (n_snapshots .lt. 0) then
    write(*,*) 'ERROR: ptrace_gc: n_snapshots must be >= 0, got ', n_snapshots
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  if (stall_rate_fraction .le. 0.d0 .or. stall_rate_fraction .ge. 1.d0 .or. &
      stall_min_lost .lt. 0.d0 .or. stall_min_lost .gt. 1.d0 .or. &
      stall_min_time .lt. 0.d0 .or. stall_min_time .gt. 1.d0 .or. stall_window .lt. 0.d0) then
    write(*,*) 'ERROR: ptrace_gc: stall_rate_fraction must be in (0, 1), stall_min_lost and ', &
      'stall_min_time in [0, 1] and stall_window >= 0; got ', stall_rate_fraction, &
      stall_min_lost, stall_min_time, stall_window
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  if (trim(field_mode) .ne. 'static' .and. trim(field_mode) .ne. 'evolving') then
    write(*,*) "ERROR: ptrace_gc: field_mode must be 'static' or 'evolving', got ", trim(field_mode)
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  if (trim(initialiser) .ne. 'markers' .and. trim(initialiser) .ne. 'current_pdf_simple') then
    write(*,*) "ERROR: ptrace_gc: initialiser must be 'markers' or 'current_pdf_simple', got ", &
      trim(initialiser)
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  if (trim(initialiser) .eq. 'current_pdf_simple' .and. E_kin_eV(1) .le. 0.d0) then
    write(*,*) 'ERROR: ptrace_gc: current_pdf_simple gives every marker E_kin_eV(1), which is ', &
      E_kin_eV(1), '; set E_kin_eV = <energy in eV>'
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  if (pdf_n_sub .lt. 1) then
    write(*,*) 'ERROR: ptrace_gc: pdf_n_sub must be >= 1, got ', pdf_n_sub
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
end if
call MPI_Bcast(field_mode, len(field_mode), MPI_CHARACTER, 0, MPI_COMM_WORLD, ierr)
! Every rank takes the initialiser's branch below (its broadcasts are collective).
call MPI_Bcast(initialiser, len(initialiser), MPI_CHARACTER, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(restart_index, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(hold_last_field, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(t_span, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(dt, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(diag_step, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(snapshot_step, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(n_snapshots, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(stop_when_stalled, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(stall_rate_fraction, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(stall_min_lost, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(stall_min_time, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(stall_window, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(mass, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(n_markers, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(R0, n_markers, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(Z0, n_markers, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(phi0, n_markers, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(E_kin_eV, n_markers, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(cos_pitch, n_markers, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
call MPI_Bcast(charge, n_markers, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)

! --- initialiser: fill R0, Z0, phi0, ... on rank 0, then share them ---
! Before the fields are read: sampling imports its own restart, which
! resets JOREK's globals (time, element search tree) -- reading the fields
! next sets them back for the trace. Every restart of a run shares its grid.
if (trim(initialiser) .eq. 'current_pdf_simple') then
  if (my_rank .eq. 0) then
    call sample_current_pdf(n_markers, pdf_n_sub, seed, &
                            R0(1:n_markers), Z0(1:n_markers), phi0(1:n_markers))
    E_kin_eV(1:n_markers)  = E_kin_eV(1)
    cos_pitch(1:n_markers) = cos_pitch(1)
    charge(1:n_markers)    = charge(1)
  end if
  call MPI_Bcast(R0, n_markers, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(Z0, n_markers, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(phi0, n_markers, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(E_kin_eV, n_markers, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(cos_pitch, n_markers, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(charge, n_markers, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
end if

! --- fields: reading them sets sim%time to the first restart's time ---
if (trim(field_mode) .eq. 'static') then
  field_reader = event(read_jorek_fields_interp_linear(i=-1))
else
  ! Never stop_at_end: that MPI_Aborts at the last restart with no outputs
  ! written. Without hold_last_field the stop event below ends the run there.
  field_reader = event(read_jorek_fields_interp_linear(i=restart_index, stop_at_end=.false.))
end if
call with(sim, field_reader)
t_start = sim%time
if (my_rank .eq. 0) write(*,'(A,ES14.6,A)') 'ptrace_gc: start time ', t_start, ' s'

! --- which JOREK step each restart is: where to stop, and to name snapshots ---
if (my_rank .eq. 0) then
  call read_restart_table(restart_index, n_rst, rst_steps, rst_times)
  do k = 1, n_rst
    write(*,'(A,I0,A,ES14.6,A)') 'ptrace_gc: restart step ', rst_steps(k), ' at t = ', rst_times(k), ' s'
  end do
end if
call MPI_Bcast(n_rst, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
if (my_rank .ne. 0) allocate(rst_steps(n_rst), rst_times(n_rst))
if (n_rst .gt. 0) then
  call MPI_Bcast(rst_steps, n_rst, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(rst_times, n_rst, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
end if

! --- when to stop: at the last linked restart, or t_span on if given ---
if (t_span .le. 0.d0) then
  t_stop = t_start
  if (n_rst .gt. 0) t_stop = max(rst_times(n_rst), t_start)
  if (my_rank .eq. 0) then
    if (t_stop .le. t_start + SNAP_TICK) then
      write(*,*) 'ERROR: ptrace_gc: nothing to trace -- t_span = 0 runs to the last linked ', &
        'restart, and there is none after the first. Set ptrace_end_step past ', &
        'ptrace_start_step, or t_span > 0 (e.g. for static fields)'
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if
    write(*,'(A,I0,A,ES12.5,A)') 'ptrace_gc: tracing to the last restart, step ', &
      rst_steps(n_rst), ', t = ', t_stop, ' s'
  end if
else
  t_stop = t_start + t_span
end if
if (t_span .gt. 0.d0 .and. trim(field_mode) .eq. 'evolving' .and. .not. hold_last_field &
    .and. n_rst .gt. 0) then
  if (t_stop .gt. rst_times(n_rst) + SNAP_TICK) then
    t_stop = max(rst_times(n_rst), t_start)
    if (my_rank .eq. 0) write(*,'(A,ES12.5,A,I0,A,ES12.5,A,ES12.5,A,ES12.5,A)') &
      'ptrace_gc: NOTE: stopped early -- t_span runs to t = ', t_start + t_span, &
      ' s, but the last restart (step ', rst_steps(n_rst), ') is at t = ', rst_times(n_rst), &
      ' s; traced ', t_stop - t_start, ' of ', t_span, &
      ' s. Raise ptrace_end_step, or set hold_last_field = .true. to go on in its frozen field.'
  end if
end if

! --- how often to write a snapshot, if not given: from the traced time ---
! t_start and t_stop are the same on every rank, so this is too.
if (snapshot_step .le. 0.d0 .and. n_snapshots .gt. 0 .and. t_stop .gt. t_start + SNAP_TICK) then
  snapshot_step = tidy_step((t_stop - t_start) / n_snapshots)
  if (my_rank .eq. 0) write(*,'(A,ES10.3,A,I0,A,I0,A)') 'ptrace_gc: a snapshot every ', &
    snapshot_step, ' s: ', int((t_stop - t_start) / snapshot_step * (1.d0 + 1.d-9)) + 1, &
    ' over the trace (n_snapshots = ', n_snapshots, ')'
end if

! --- markers, round-robin over ranks ---
sim%groups(1)%mass = mass
n_local = 0
do k = 1, n_markers
  if (mod(k-1, n_ranks) .eq. my_rank) n_local = n_local + 1
end do
allocate(particle_gc_relativistic::sim%groups(1)%particles(n_local))

rest_energy_eV = mass * ATOMIC_MASS_UNIT * SPEED_OF_LIGHT**2 / EL_CHG
j = 0
do k = 1, n_markers
  if (mod(k-1, n_ranks) .ne. my_rank) cycle
  j = j + 1
  select type (p => sim%groups(1)%particles(j))
  type is (particle_gc_relativistic)
    p%q      = int(charge(k), kind=1)
    p%x      = [R0(k), Z0(k), phi0(k)]
    call find_RZ(sim%fields%node_list, sim%fields%element_list, &
                 R0(k), Z0(k), &                                        ! inputs
                 p%x(1), p%x(2), p%i_elm, p%st(1), p%st(2), ifail)      ! outputs
    if (p%i_elm .le. 0) then
      write(*,'(A,I0,A,2F10.5,A)') 'WARNING: ptrace_gc: marker ', k, ' at (R,Z) = (', &
        R0(k), Z0(k), ') is outside the grid; it is reported lost from the start'
      p%i_elm = 0
      cycle
    end if
    if (abs(cos_pitch(k)) .gt. 1.d0) then
      write(*,'(A,I0,A,ES12.4,A)') 'ERROR: ptrace_gc: marker ', k, ' has |cos_pitch| > 1: ', cos_pitch(k), &
        ' (cos_pitch has no default: set it)'
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if
    marker = p
    marker = relativistic_gc_momenta_from_E_cospitch(marker, E_kin_eV(k) + rest_energy_eV, &
                                                     cos_pitch(k), mass, sim%fields, sim%time)
    p%p = marker%p
  end select
end do

! --- events: diagnostics from the start, stop at t_stop ---
diag = write_particle_diagnostics(filename='ptrace_diag.h5', &
                                  only=[1,2,3,4,6,8,11,12,13,14,15]) ! e, k, mu, psi_n, p_phi, lost, theta, phi, R, Z, i_elm
events = [field_reader, &
          event(diag, start=t_start, step=diag_step), &
          event(stop_action(), start=t_stop)]
call with(sim, events, at=sim%time)

! --- stopping early: what there is to watch, and the count at the start ---
allocate(has_left(n_local))
has_left = .false.
n_bnd = 0
n_samples = 0
max_loss_rate = 0.d0
t_check = t_start
! (A trace of no length has nothing to stall in.)
if (t_stop .le. t_start + SNAP_TICK) stop_when_stalled = .false.
if (stop_when_stalled) then
  if (stall_window .le. 0.d0) stall_window = (t_stop - t_start) / 10.d0
  if (my_rank .eq. 0) call read_boundary(BOUNDARY_FILE, n_bnd, bnd_R, bnd_Z)
  call MPI_Bcast(n_bnd, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
  if (my_rank .ne. 0) allocate(bnd_R(n_bnd), bnd_Z(n_bnd))
  if (n_bnd .gt. 0) then
    call MPI_Bcast(bnd_R, n_bnd, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
    call MPI_Bcast(bnd_Z, n_bnd, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
  end if
  k = int(SAMPLES_PER_WINDOW * (t_stop - t_start) / stall_window) + 8
  allocate(sample_t(k), sample_n(k))
  if (my_rank .eq. 0) then
    write(*,'(A,ES10.3,A,F6.2,A,F6.2,A,F6.2,A)') &
      'ptrace_gc: stop_when_stalled: the loss rate over ', stall_window, ' s, against ', &
      100.d0 * stall_rate_fraction, ' % of its peak, once ', 100.d0 * stall_min_lost, &
      ' % of the markers has left or ', 100.d0 * stall_min_time, ' % of the trace has passed'
    if (n_bnd .gt. 0) then
      write(*,'(A,I0,A,A)') 'ptrace_gc: stop_when_stalled: a marker has left once outside the ', &
        n_bnd, '-point outline in ', BOUNDARY_FILE
    else
      write(*,'(A,A,A)') 'ptrace_gc: stop_when_stalled: no ', BOUNDARY_FILE, &
        ', so a marker has left once it is off the grid'
    end if
  end if
end if
call check_stop()

! Snapshots are scheduled here rather than as an event: their filename
! carries the closest restart's step, which an io_action cannot build.
n_snap = 0
t_snap = t_start
if (snapshot_step .gt. 0.d0) then
  call write_snapshot()
  n_snap = 1
  t_snap = t_start + snapshot_step
end if

! --- push ---
do while (.not. sim%stop_now)
  target_time = next_event_at(sim, events)
  if (snapshot_step .gt. 0.d0 .and. t_snap .lt. target_time - SNAP_TICK) target_time = t_snap
  call push_all(sim, dt, target_time)
  sim%time = target_time
  call with(sim, events, at=sim%time)
  if (snapshot_step .gt. 0.d0 .and. abs(sim%time - t_snap) .le. SNAP_TICK) then
    call write_snapshot()
    n_snap = n_snap + 1
    t_snap = t_start + n_snap * snapshot_step  ! from t_start, so steps don't accumulate round-off
  end if
  call check_stop()
end do

n_lost = 0
select type (particles => sim%groups(1)%particles)
type is (particle_gc_relativistic)
  n_lost = count(particles(:)%i_elm .le. 0)
end select
call MPI_Allreduce(MPI_IN_PLACE, n_lost, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
if (my_rank .eq. 0) write(*,'(A,ES12.5,A,I0,A,I0,A)') 'ptrace_gc: done at t = ', sim%time, &
  ' s, ', n_lost, ' of ', n_markers, ' markers lost'

call write_simulation_hdf5(sim, 'part_restart.h5')
call sim%finalize

contains

!> Push every marker from sim%time to target_time with fixed-step RK4.
!> A lost marker (i_elm <= 0) is left where it left the grid.
!> Scheduled dynamically: a lost marker costs nothing and the others vary
!> (element searches), so a static split leaves threads idle. Each marker is
!> pushed on its own from the same sim%time, so which thread takes it
!> changes nothing in the result.
subroutine push_all(sim, dt, target_time)
  type(particle_sim), intent(inout) :: sim
  real*8, intent(in)                :: dt, target_time
  integer :: i
  real*8  :: local_time, local_dt
  !$omp parallel do default(none) firstprivate(dt, target_time) &
  !$omp private(i, local_time, local_dt) shared(sim) schedule(dynamic)
  do i = 1, size(sim%groups(1)%particles)
    select type (gc => sim%groups(1)%particles(i))
    type is (particle_gc_relativistic)
      local_time = sim%time
      local_dt   = min(dt, target_time - local_time)
      do while ((target_time - local_time .gt. 0.d0) .and. (gc%i_elm .gt. 0))
        call runge_kutta_fixed_dt_gc_push_jorek(sim%fields, local_time, local_dt, &
                                                sim%groups(1)%mass, gc)
        local_time = local_time + local_dt
        local_dt   = min(dt, target_time - local_time)
      end do
    end select
  end do
  !$omp end parallel do
end subroutine push_all

!> n markers with a restart's toroidal current density as their pdf in
!> space (see 'current_pdf_simple' at the top): PDF_RESTART if it is there
!> (ashen links ptrace_pdf_step's restart as it), else jorek_restart.h5 (the
!> first restart). Per unit R-Z area the density is R*|j_phi| ~ |zj|,
!> zj = Delta* psi: each of n_sub x n_sub cells per grid element gets its
!> n = 0 zj times its area, keeping only the part along the net current;
!> markers are drawn from that table, uniform in (s, t) within their cell
!> and in phi. Imports the restart into lists of its own, which resets
!> JOREK's globals: call it before the fields are read.
subroutine sample_current_pdf(n, n_sub, seed, R, Z, phi)
  use data_structure,     only: type_node_list, type_element_list
  use mod_import_restart, only: import_hdf5_restart
  use mod_interp,         only: interp, interp_RZ
  use mod_model_settings, only: var_zj
  use constants,          only: TWOPI
  integer, intent(in)  :: n, n_sub, seed
  real*8,  intent(out) :: R(n), Z(n), phi(n)
  character(len=*), parameter :: PDF_RESTART = 'jorek_pdf.h5'
  type(type_node_list),    pointer :: nodes
  type(type_element_list), pointer :: elements
  real*8, allocatable  :: cell(:), cdf(:)
  integer, allocatable :: seed_arr(:)
  character(len=32) :: filename
  logical :: exists
  integer :: n_elm, n_cells, i_elm, i, j, k, c, lo, hi, mid, n_seed, rst_err, pdf_step
  real*8  :: s, t, zj, zj_s, zj_t, zj_st, zj_ss, zj_tt
  real*8  :: RR, R_s, R_t, ZZ, Z_s, Z_t, net, along, against, u(4), pdf_time, t_norm

  filename = 'jorek_restart.h5'
  inquire(file=PDF_RESTART, exist=exists)
  if (exists) filename = PDF_RESTART
  ! The element list is a large fixed-size array: on the heap, not the stack.
  allocate(nodes, elements)
  call import_hdf5_restart(nodes, elements, trim(filename), 0, rst_err)
  if (rst_err .ne. 0) then
    write(*,*) 'ERROR: ptrace_gc: current_pdf_simple: cannot read ', trim(filename)
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  t_norm = sqrt(MU_ZERO * ATOMIC_MASS_UNIT * central_mass * central_density * 1.d20)
  call read_restart_stamp(trim(filename), t_norm, pdf_step, pdf_time)
  write(*,'(A,I0,A,ES12.5,A,A,A)') 'ptrace_gc: current_pdf_simple: current profile of step ', &
    pdf_step, ' (t = ', pdf_time, ' s, ', trim(filename), ')'

  n_elm   = elements%n_elements
  n_cells = n_elm * n_sub * n_sub
  allocate(cell(n_cells), cdf(n_cells))
  c = 0
  do i_elm = 1, n_elm
    do i = 1, n_sub
      do j = 1, n_sub
        c = c + 1
        s = (i - 0.5d0) / n_sub
        t = (j - 0.5d0) / n_sub
        ! i_harm = 1: the n = 0 (axisymmetric) part of zj
        call interp(nodes, elements, i_elm, var_zj, 1, s, t, zj, zj_s, zj_t, zj_st, zj_ss, zj_tt)
        call interp_RZ(nodes, elements, i_elm, s, t, RR, R_s, R_t, ZZ, Z_s, Z_t)
        cell(c) = zj * abs(R_s * Z_t - R_t * Z_s) / (n_sub * n_sub)
      end do
    end do
  end do

  ! Only the current along the net current: its sign is the plasma current's.
  net = sum(cell)
  cell = sign(1.d0, net) * cell
  along   = sum(max(cell, 0.d0))
  against = sum(max(-cell, 0.d0))
  if (along .le. 0.d0) then
    write(*,*) 'ERROR: ptrace_gc: current_pdf_simple: no toroidal current on the grid'
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  write(*,'(A,I0,A,I0,A,F6.2,A)') 'ptrace_gc: current_pdf_simple: ', n, ' markers from ', &
    n_cells, ' cells; ', 100.d0 * against / (along + against), &
    ' % of |current| runs against the net current and gets none'
  cdf(1) = max(cell(1), 0.d0)
  do c = 2, n_cells
    cdf(c) = cdf(c-1) + max(cell(c), 0.d0)
  end do

  call random_seed(size=n_seed)
  allocate(seed_arr(n_seed))
  seed_arr = seed + 104729 * [(k, k = 0, n_seed - 1)]
  call random_seed(put=seed_arr)
  do k = 1, n
    call random_number(u)
    ! the first cell whose cumulative share reaches u(1)
    lo = 1
    hi = n_cells
    do while (lo .lt. hi)
      mid = (lo + hi) / 2
      if (cdf(mid) .lt. u(1) * cdf(n_cells)) then
        lo = mid + 1
      else
        hi = mid
      end if
    end do
    c = lo - 1                                   ! 0-based: element, then i, then j
    i_elm = c / (n_sub * n_sub) + 1
    s = (mod(c, n_sub * n_sub) / n_sub + u(2)) / n_sub
    t = (mod(c, n_sub) + u(3)) / n_sub
    call interp_RZ(nodes, elements, i_elm, s, t, RR, R_s, R_t, ZZ, Z_s, Z_t)
    R(k)   = RR
    Z(k)   = ZZ
    phi(k) = TWOPI * u(4)
  end do
  deallocate(nodes, elements)
end subroutine sample_current_pdf

!> Write every particle to part_restart_s<step>_t<time>.h5: <step> is the
!> JOREK step of the restart closest in time (static fields: the first
!> restart's, the field used), <time> the simulation time [s].
!> Collective over MPI ranks, like write_simulation_hdf5 itself.
subroutine write_snapshot()
  character(len=128) :: fname
  integer :: closest
  if (n_rst .gt. 0) then
    closest = minloc(abs(rst_times(1:n_rst) - sim%time), dim=1)
    if (trim(field_mode) .eq. 'static') closest = 1
    write(fname,'(A,I6.6,A,ES12.6,A)') 'part_restart_s', rst_steps(closest), '_t', sim%time, '.h5'
  else
    write(fname,'(A,ES12.6,A)') 'part_restart_t', sim%time, '.h5'
  end if
  call write_simulation_hdf5(sim, trim(fname))
end subroutine write_snapshot

!> The JOREK step and time of every linked restart: the sequence
!> jorek<i>.h5 from first_index -- ashen links it for static fields too, so
!> the last one says where to stop -- or, without one, jorek_restart.h5
!> alone. Looked for at 6 digits, then 5: ashen links both, and JOREK
!> versions differ in which one the field reader opens (newer ones name it
!> in mod_import_restart's rst_file_ind_fmt, which older ones lack).
subroutine read_restart_table(first_index, n, steps, times)
  integer, intent(in)                :: first_index
  integer, intent(out)               :: n
  integer, allocatable, intent(out)  :: steps(:)
  real*8,  allocatable, intent(out)  :: times(:)
  integer, parameter :: MAX_RESTARTS = 100000
  integer, allocatable :: all_steps(:)
  real*8,  allocatable :: all_times(:)
  character(len=80) :: fname
  logical :: exists
  integer :: i
  real*8  :: t_norm

  ! 1 JOREK time unit in seconds, as mod_fields_linear converts t_now
  t_norm = sqrt(MU_ZERO * ATOMIC_MASS_UNIT * central_mass * central_density * 1.d20)
  allocate(all_steps(MAX_RESTARTS), all_times(MAX_RESTARTS))
  n = 0
  do i = first_index, first_index + MAX_RESTARTS - 1
    write(fname, '(A,I6.6,A)') 'jorek', i, '.h5'
    inquire(file=trim(fname), exist=exists)
    if (.not. exists) then
      write(fname, '(A,I5.5,A)') 'jorek', i, '.h5'
      inquire(file=trim(fname), exist=exists)
    end if
    if (.not. exists) exit
    n = n + 1
    call read_restart_stamp(trim(fname), t_norm, all_steps(n), all_times(n))
  end do
  if (n .eq. 0) then
    inquire(file='jorek_restart.h5', exist=exists)
    if (exists) then
      n = 1
      call read_restart_stamp('jorek_restart.h5', t_norm, all_steps(1), all_times(1))
    end if
  end if
  allocate(steps(n), times(n))
  steps = all_steps(1:n)
  times = all_times(1:n)
end subroutine read_restart_table

!> End the trace early when there is nothing more to learn from it (see
!> "Stopping early" at the top). Called wherever the push loop has stopped
!> anyway, so it adds no stopping time of its own. Collective: every rank
!> calls it at the same times and reaches the same answer, the counts being
!> summed over all ranks.
subroutine check_stop()
  integer :: i, j, counts(2)
  real*8  :: loss_rate
  if (sim%stop_now) return

  ! counts(1): markers that have left, by the stall rule's meaning;
  ! counts(2): markers off the grid, which nothing pushes any more
  counts = 0
  select type (particles => sim%groups(1)%particles)
  type is (particle_gc_relativistic)
    do i = 1, size(particles)
      if (particles(i)%i_elm .le. 0) then
        counts(2) = counts(2) + 1
        has_left(i) = .true.
      else if (n_bnd .gt. 0 .and. .not. has_left(i)) then
        has_left(i) = .not. inside_boundary(particles(i)%x(1), particles(i)%x(2))
      end if
    end do
  end select
  counts(1) = count(has_left)
  call MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
  n_left = counts(1)

  if (counts(2) .ge. n_markers) then
    sim%stop_now = .true.
    if (my_rank .eq. 0) write(*,'(A,ES12.5,A,F6.2,A)') &
      'ptrace_gc: NOTE: every marker is off the grid -- stopped at t = ', sim%time, ' s, ', &
      100.d0 * (sim%time - t_start) / max(t_stop - t_start, SNAP_TICK), &
      ' % of the way through the trace'
    return
  end if

  if (.not. stop_when_stalled) return
  if (sim%time .lt. t_check - SNAP_TICK) return
  if (n_samples .ge. size(sample_t)) return
  n_samples = n_samples + 1
  sample_t(n_samples) = sim%time
  sample_n(n_samples) = n_left
  t_check = sim%time + stall_window / SAMPLES_PER_WINDOW

  ! the loss rate over the last stall_window: against the newest sample at
  ! least that far back -- none until a whole window has been traced
  j = 0
  do i = n_samples - 1, 1, -1
    if (sample_t(i) .le. sim%time - stall_window + SNAP_TICK) then
      j = i
      exit
    end if
  end do
  if (j .eq. 0) return
  loss_rate = (sample_n(n_samples) - sample_n(j)) / (sample_t(n_samples) - sample_t(j))
  max_loss_rate = max(max_loss_rate, loss_rate)

  ! Never while no marker has left in any window: that is a trace whose
  ! losses have not begun, not one whose losses are over.
  if (max_loss_rate .le. 0.d0) return
  ! Not before either gate: enough markers gone, or enough of the trace done.
  if (n_left .lt. stall_min_lost * n_markers .and. &
      sim%time - t_start .lt. stall_min_time * (t_stop - t_start)) return
  if (loss_rate .gt. stall_rate_fraction * max_loss_rate) return

  sim%stop_now = .true.
  if (my_rank .eq. 0) write(*,'(A,ES12.5,A,F6.2,A,F6.2,A,I0,A,I0,A)') &
    'ptrace_gc: NOTE: losses stalled -- stopped at t = ', sim%time, ' s, ', &
    100.d0 * (sim%time - t_start) / (t_stop - t_start), &
    ' % of the way through the trace: the loss rate is ', &
    100.d0 * loss_rate / max_loss_rate, ' % of its peak, with ', n_left, ' of ', n_markers, &
    ' markers gone'
end subroutine check_stop

!> Whether (R, Z) is inside the outline bnd_R, bnd_Z (closed, the last
!> point joined to the first): even-odd ray casting, the same test ashen's
!> plots use (ashen.diagnostics.particles.inside_polygon).
logical function inside_boundary(R, Z)
  real*8, intent(in) :: R, Z
  integer :: i, j
  inside_boundary = .false.
  j = n_bnd
  do i = 1, n_bnd
    if ((bnd_Z(i) .gt. Z) .neqv. (bnd_Z(j) .gt. Z)) then
      if (R .lt. bnd_R(i) + (Z - bnd_Z(i)) * (bnd_R(j) - bnd_R(i)) / (bnd_Z(j) - bnd_Z(i))) &
        inside_boundary = .not. inside_boundary
    end if
    j = i
  end do
end function inside_boundary

!> The outline in filename -- a line with n, then n lines of R Z [m] -- or
!> n = 0 if there is no such file.
subroutine read_boundary(filename, n, R, Z)
  character(len=*), intent(in)      :: filename
  integer, intent(out)              :: n
  real*8, allocatable, intent(out)  :: R(:), Z(:)
  logical :: exists
  integer :: unit, io, i
  n = 0
  inquire(file=filename, exist=exists)
  if (exists) then
    open(newunit=unit, file=filename, status='old', action='read', iostat=io)
    if (io .eq. 0) read(unit, *, iostat=io) n
    if (io .ne. 0 .or. n .lt. 3) then
      write(*,*) 'ERROR: ptrace_gc: ', filename, ' must start with its number of points (>= 3)'
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if
    allocate(R(n), Z(n))
    do i = 1, n
      read(unit, *, iostat=io) R(i), Z(i)
      if (io .ne. 0) then
        write(*,*) 'ERROR: ptrace_gc: ', filename, ': cannot read R Z of point ', i
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
      end if
    end do
    close(unit)
  else
    allocate(R(0), Z(0))
  end if
end subroutine read_boundary

!> raw rounded up to 1, 2 or 5 times a power of ten: a snapshot spacing
!> that puts the snapshots at round times. raw > 0.
pure function tidy_step(raw) result(step)
  real*8, intent(in) :: raw
  real*8 :: step, base, mantissa
  real*8, parameter :: SLACK = 1.d0 + 1.d-9   ! 2.0000000001 is still 2
  base     = 10.d0**floor(log10(raw))
  mantissa = raw / base                       ! in [1, 10)
  if (mantissa .le. 1.d0 * SLACK) then
    step = base
  else if (mantissa .le. 2.d0 * SLACK) then
    step = 2.d0 * base
  else if (mantissa .le. 5.d0 * SLACK) then
    step = 5.d0 * base
  else
    step = 10.d0 * base
  end if
end function tidy_step

!> One restart's step (index_now) and time (t_now, converted to seconds).
subroutine read_restart_stamp(filename, t_norm, step, time)
  use hdf5
  use hdf5_io_module, only: HDF5_integer_reading, HDF5_real_reading
  character(len=*), intent(in) :: filename
  real*8, intent(in)           :: t_norm
  integer, intent(out)         :: step
  real*8, intent(out)          :: time
  integer(HID_T) :: file_id
  integer        :: hdferr
  call h5open_f(hdferr)
  call h5fopen_f(filename, H5F_ACC_RDONLY_F, file_id, hdferr)
  if (hdferr .ne. 0) then
    write(*,*) 'WARNING: ptrace_gc: cannot open ', filename, ' to read its step'
    step = 0
    time = -1.d0
    return
  end if
  call HDF5_integer_reading(file_id, step, 'index_now')
  call HDF5_real_reading(file_id, time, 't_now')
  call h5fclose_f(file_id, hdferr)
  time = time * t_norm
end subroutine read_restart_stamp

end program ptrace_gc
